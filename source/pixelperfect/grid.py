"""Finite lattice proposals, joint phase fitting and gated 1-D grid drift.

This is an engineering estimator. Fourier and autocorrelation proposals use
the same signal and are deliberately not counted as independent votes.
"""
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

from .features import FeatureData


@dataclass
class GridCandidate:
    sx: float
    sy: float
    phase_x: float
    phase_y: float
    x_lines: np.ndarray
    y_lines: np.ndarray
    edge_score: float = 1.0
    support: float = 0.0
    strip_agreement: float = 0.0
    warped: bool = False
    warp_penalty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def make_lines(length: int, spacing: float, phase: float, count: int | None = None) -> np.ndarray:
    """Cover the whole input, merging boundary remnants narrower than 0.2 s.

    Coordinates are continuous source pixel boundaries. For exact target size,
    outer boundaries stay fixed and interior boundaries get one bounded phase
    shift; no fragment is removed in that mode.
    """
    if length < 1 or spacing <= 0 or not np.isfinite(spacing + phase):
        raise ValueError("invalid grid geometry")
    if count is not None:
        if count < 1 or count > length:
            raise ValueError("target grid count must be between 1 and source size")
        base = length / count
        shift = ((phase + base / 2) % base) - base / 2
        shift = float(np.clip(shift, -.3 * base, .3 * base))
        return np.r_[0., np.arange(1, count) * base + shift, float(length)]
    phase = phase % spacing
    first = phase if phase > 1e-6 else spacing
    inner = np.arange(first, length - 1e-6, spacing, dtype=float)
    if inner.size and inner[0] < .2 * spacing:
        inner = inner[1:]
    if inner.size and length - inner[-1] < .2 * spacing:
        inner = inner[:-1]
    return np.r_[0., inner, float(length)]


def _phase_fit(positions, weights, spacing, profiles=None, active=None):
    if len(positions) == 0:
        return 0., 1., 0., 0.
    # A uniform search plus actual edge residues handles noninteger scales and
    # exact nearest-neighbour data without rounding phase to a pixel.
    phases = np.unique(np.r_[np.linspace(0, spacing, min(96, max(12, int(spacing * 4))), endpoint=False),
                              np.mod(positions[:256], spacing)])
    distance = np.abs(((positions[:, None] - phases + spacing / 2) % spacing) - spacing / 2) / spacing
    losses = np.minimum((distance / .22) ** 2, 1.)
    scores = (weights[:, None] * losses).sum(axis=0) / max(weights.sum(), 1e-9)
    index = int(np.argmin(scores))
    phase, error = float(phases[index]), float(scores[index])
    support = float(np.average(distance[:, index] < max(.08, .7 / spacing), weights=weights))
    agreements = []
    if profiles is not None:
        for p in profiles[active]:
            peaks, _ = find_peaks(p, prominence=max(.05, float(p.max()) * .12))
            if len(peaks) < 2:
                continue
            d = np.abs(((peaks - phase + spacing / 2) % spacing) - spacing / 2) / spacing
            agreements.append(float(np.average(d < max(.1, .8 / spacing), weights=p[peaks])))
    agreement = float(np.mean(agreements)) if agreements else 0.
    return phase, error, support, agreement


def _axis_candidates(profile, profiles, active, positions, weights, minimum, maximum, limit):
    if len(positions) < 3 or maximum < minimum:
        return [], {"reason": "insufficient directional edges", "considered": []}
    n = len(profile)
    centered = profile.astype(float) - float(np.mean(profile))
    spectrum = np.abs(np.fft.rfft(centered, n=2 * n)) ** 2
    ac = np.fft.irfft(spectrum, n=2 * n)[:n]
    if ac[0] <= 1e-12:
        return [], {"reason": "zero periodic energy", "considered": []}
    ac /= ac[0]
    ac_peaks, _ = find_peaks(ac)
    ac_peaks = [int(p) for p in ac_peaks if minimum <= p <= maximum]
    ac_peaks.sort(key=lambda p: (-ac[p], p))
    fourier = np.abs(np.fft.rfft(centered))
    bins, _ = find_peaks(fourier)
    bins = sorted((int(k) for k in bins if k > 0 and minimum <= n / k <= maximum), key=lambda k: -fourier[k])[:6]
    gaps = np.diff(positions)
    # Adjacent-gap votes and missing-boundary submultiples are another proposal
    # family. They are not extra confidence votes for FFT/ACF peaks.
    histogram = {}
    for gap, w in zip(gaps, np.minimum(weights[:-1], weights[1:])):
        key = round(float(gap) * 2) / 2
        if minimum <= key <= maximum:
            histogram[key] = histogram.get(key, 0.) + float(w)
    gap_seeds = sorted(histogram, key=lambda s: (-histogram[s], s))[:5]
    seeds = list(dict.fromkeys(gap_seeds + ac_peaks[:5] + [n / k for k in bins]))
    proposals = {}
    def add(s, family):
        if minimum <= s <= maximum:
            proposals.setdefault(round(float(s), 5), set()).add(family)
    for s in seeds:
        add(s, "base")
        add(s / 2, "half")
        add(s * 2, "double")
        add(s - .25, "nearby")
        add(s + .25, "nearby")
    considered = []
    for s, families in proposals.items():
        phase, error, support, agreement = _phase_fit(positions, weights, s, profiles, active)
        autocorr = max(0., float(np.interp(s, np.arange(n), ac)))
        # Occupancy is a weak preference, not a demand for visible boundaries:
        # the reconstruction stage is responsible for choosing among harmonics.
        aligned = np.abs(((positions - phase + s / 2) % s) - s / 2) < max(.8, .1 * s)
        bins_hit = np.unique(np.round((positions[aligned] - phase) / s)).size
        occupancy = min(1., bins_hit / max(n / s - 1, 1.))
        quality = error + .14 * (1 - autocorr) + .12 * (1 - occupancy) + .1 * (1 - agreement)
        considered.append({"spacing": s, "phase": phase, "edge_error": error,
                           "support": support, "agreement": agreement,
                           "autocorrelation": autocorr, "occupancy": occupancy,
                           "quick_score": quality, "families": sorted(families)})
    considered.sort(key=lambda c: (c["quick_score"], c["spacing"]))
    # Refine a few plausible scales continuously. Pixel-boundary observations
    # are integer coordinates, but their repeated mean spacing need not be.
    refined = []
    for seed in considered[:3]:
        spacing = seed["spacing"]
        radius = min(.55, spacing * .06)
        lo, hi = max(minimum, spacing - radius), min(maximum, spacing + radius)
        if hi - lo < 1e-5:
            continue
        fit = minimize_scalar(lambda s: _phase_fit(positions, weights, s)[1],
                              bounds=(lo, hi), method="bounded", options={"maxiter": 14, "xatol": .002})
        s = float(fit.x)
        phase, error, support, agreement = _phase_fit(positions, weights, s, profiles, active)
        autocorr = max(0., float(np.interp(s, np.arange(n), ac)))
        aligned = np.abs(((positions - phase + s / 2) % s) - s / 2) < max(.8, .1 * s)
        occupancy = min(1., np.unique(np.round((positions[aligned] - phase) / s)).size / max(n / s - 1, 1.))
        quality = error + .14 * (1 - autocorr) + .12 * (1 - occupancy) + .1 * (1 - agreement)
        refined.append(dict(spacing=s, phase=phase, edge_error=error, support=support,
                            agreement=agreement, autocorrelation=autocorr, occupancy=occupancy,
                            quick_score=quality, families=["continuous refinement"]))
    considered.extend(refined)
    considered.sort(key=lambda c: (c["quick_score"], c["spacing"]))
    if not considered:
        return [], {"reason": "no scales in range", "considered": []}
    best = considered[0]
    # Keep harmonics explicitly in the finite shortlist rather than allowing
    # six neighbouring subpixel proposals to consume all available slots.
    selected = [best]
    for wanted in (best["spacing"] / 2, best["spacing"] * 2):
        matches = [c for c in considered if abs(c["spacing"] - wanted) < .04]
        if matches and len(selected) < limit:
            selected.append(min(matches, key=lambda c: (abs(c["spacing"] - wanted), c["quick_score"])))
    for c in considered:
        if len(selected) >= limit:
            break
        if all(abs(c["spacing"] - k["spacing"]) > max(.35, .035 * c["spacing"]) for k in selected):
            selected.append(c)
    # A smooth edge or sparse object boundary does not establish a lattice.
    # Merely having many peaks is not periodic evidence: noise can align its
    # few aggregate maxima with a tiny integer lattice by chance. Require
    # cross-strip agreement, with stronger agreement when periodicity is weak.
    meaningful = best["support"] >= .45 and (
        (best["autocorrelation"] > .08 and best["agreement"] > .45) or
        (best["agreement"] >= .68 and len(positions) >= 5))
    if not meaningful:
        return [], {"reason": "weak periodic evidence", "considered": considered, "selected": selected}
    return selected, {"reason": "periodic proposals", "considered": considered, "selected": selected,
                      "spectral_evidence_family": "FFT and autocorrelation share one projection"}


def generate_candidates(features: FeatureData, shape_hw, config) -> list[GridCandidate]:
    height, width = map(int, shape_hw[:2])
    manual = config.pixel_size is not None or config.target_size is not None
    targets = config.target_size
    if manual:
        if targets is not None:
            sx, sy = width / targets[0], height / targets[1]
        elif np.isscalar(config.pixel_size):
            sx = sy = float(config.pixel_size)
        else:
            sx, sy = map(float, config.pixel_size)
        axes = []
        for s, pos, weight, profiles, active in (
            (sx, features.edge_positions_x, features.edge_weights_x, features.profiles_x, features.active_x),
            (sy, features.edge_positions_y, features.edge_weights_y, features.profiles_y, features.active_y)):
            phase, error, support, agreement = _phase_fit(pos, weight, s, profiles, active)
            axes.append([dict(spacing=s, phase=phase, edge_error=error, support=support,
                              agreement=agreement, quick_score=error, families=["user constraint"])])
        xs, ys = axes
        dx, dy = {"selected": xs}, {"selected": ys}
    else:
        minimum = max(1.25, float(config.min_pixel_size))
        xs, dx = _axis_candidates(features.profile_x, features.profiles_x, features.active_x,
                                 features.edge_positions_x, features.edge_weights_x,
                                 minimum, min(config.max_pixel_size, width / 2), config.max_axis_candidates)
        ys, dy = _axis_candidates(features.profile_y, features.profiles_y, features.active_y,
                                 features.edge_positions_y, features.edge_weights_y,
                                 minimum, min(config.max_pixel_size, height / 2), config.max_axis_candidates)
        if not xs or not ys:
            return []
    candidates = []
    for x_seed in xs:
        for y in ys:
            x = x_seed.copy()
            sx, sy = x["spacing"], y["spacing"]
            mismatch = abs(float(np.log(sx / sy)))
            if config.square and not manual:
                if mismatch > .12:
                    continue
                common = (sx + sy) / 2
                sx = sy = common
                xp = _phase_fit(features.edge_positions_x, features.edge_weights_x, sx,
                                features.profiles_x, features.active_x)
                yp = _phase_fit(features.edge_positions_y, features.edge_weights_y, sy,
                                features.profiles_y, features.active_y)
                x = {**x, "phase": xp[0], "edge_error": xp[1], "support": xp[2], "agreement": xp[3]}
                y2 = {**y, "phase": yp[0], "edge_error": yp[1], "support": yp[2], "agreement": yp[3]}
            else:
                y2 = y
            edge = (x["edge_error"] + y2["edge_error"]) / 2
            quick = (x["quick_score"] + y2["quick_score"]) / 2 + .2 * min(mismatch, 2.)
            metadata = {"quick_score": float(quick), "axis_candidates": {"x": dx, "y": dy},
                        "manual_constraint": bool(manual), "target_size": list(targets) if targets else None,
                        "boundary_fragment_rule": "merge visible remnants narrower than 0.2 spacing; target count takes precedence",
                        "range_boundary": bool(not manual and (min(sx, sy) <= config.min_pixel_size * 1.02 or
                                                               max(sx, sy) >= config.max_pixel_size * .98))}
            candidates.append(GridCandidate(float(sx), float(sy), x["phase"], y2["phase"],
                make_lines(width, sx, x["phase"], targets[0] if targets else None),
                make_lines(height, sy, y2["phase"], targets[1] if targets else None), edge,
                min(x["support"], y2["support"]), min(x["agreement"], y2["agreement"]), metadata=metadata))
    candidates.sort(key=lambda c: (c.metadata["quick_score"], c.sx * c.sy))
    # Full reconstruction budget is small. Ensure geometrically similar scale
    # pairs include finer/coarser explanations when they exist in axis lists.
    selected = candidates[:1]
    if selected:
        anchor = selected[0]
        for factor in (.5, 2.):
            matches = [c for c in candidates if abs(c.sx / anchor.sx - factor) < .025 and
                       abs(c.sy / anchor.sy - factor) < .025]
            if matches and len(selected) < config.max_candidates:
                selected.append(matches[0])
    for c in candidates:
        if len(selected) >= config.max_candidates:
            break
        if all(abs(np.log(c.sx / k.sx)) + abs(np.log(c.sy / k.sy)) > .04 for k in selected):
            selected.append(c)
    pair_scores = [dict(sx=c.sx, sy=c.sy, phase_x=c.phase_x, phase_y=c.phase_y,
                        quick_score=c.metadata["quick_score"], shortlisted=any(c is chosen for chosen in selected))
                   for c in candidates]
    for c in selected:
        c.metadata["pair_screening"] = pair_scores
    return selected


def _warp_lines(lines, spacing, profiles, active):
    if len(lines) <= 2 or int(np.count_nonzero(active)) < 2 or spacing < 3:
        return lines.copy(), 0., False
    ps = profiles[active]
    length = ps.shape[1]
    radius = min(3., .18 * spacing)
    choices = []
    unary = []
    consensus = 0
    for line in lines[1:-1]:
        positions = np.unique(np.r_[line, np.arange(np.ceil(line - radius), np.floor(line + radius) + 1)])
        positions = positions[(positions >= .5) & (positions <= length - .5)]
        evidence = np.array([np.interp(positions, np.arange(length), p) for p in ps])
        baseline = np.array([np.interp(line, np.arange(length), p) for p in ps])
        means = evidence.mean(axis=0)
        improvements = (evidence - baseline[:, None]) > .1
        # A move must have support in at least two independently located strips.
        allowed = (improvements.sum(axis=0) >= 2) | np.isclose(positions, line)
        if np.any((improvements.sum(axis=0) >= 2) & (means > baseline.mean() + .12)):
            consensus += 1
        score = -.5 * means + .3 * ((positions - line) / spacing) ** 2
        score[~allowed] = np.inf
        choices.append(positions)
        unary.append(score)
    if consensus < max(1, int(.15 * (len(lines) - 2))):
        return lines.copy(), 0., False
    costs = unary[0].copy()
    back = []
    # Chain dynamic programming with bounded neighbouring width variation.
    for i in range(1, len(choices)):
        gaps = choices[i][None, :] - choices[i - 1][:, None]
        transition = .8 * ((gaps - spacing) / spacing) ** 2
        transition[(gaps <= .45 * spacing) | (gaps >= 1.55 * spacing)] = np.inf
        total = costs[:, None] + transition
        prev = np.argmin(total, axis=0)
        costs = unary[i] + total[prev, np.arange(len(prev))]
        back.append(prev)
    state = int(np.argmin(costs))
    if not np.isfinite(costs[state]):
        return lines.copy(), 0., False
    interior = [float(choices[-1][state])]
    for i in range(len(back) - 1, -1, -1):
        state = int(back[i][state])
        interior.append(float(choices[i][state]))
    result = np.r_[lines[0], interior[::-1], lines[-1]]
    original_support = np.mean([np.interp(lines[1:-1], np.arange(length), p).mean() for p in ps])
    new_support = np.mean([np.interp(result[1:-1], np.arange(length), p).mean() for p in ps])
    penalty = float(np.mean(((result - lines) / spacing) ** 2))
    gain = float(new_support - original_support)
    if np.any(np.diff(result) <= 0) or gain <= .035 + penalty or np.max(np.abs(result - lines)) < 1e-6:
        return lines.copy(), 0., False
    return result, penalty, True


def refine_candidate(candidate: GridCandidate, features: FeatureData, config) -> GridCandidate:
    if config.local_warp == "off":
        return candidate
    xs, xp, xchanged = _warp_lines(candidate.x_lines, candidate.sx, features.profiles_x, features.active_x)
    ys, yp, ychanged = _warp_lines(candidate.y_lines, candidate.sy, features.profiles_y, features.active_y)
    def actual_evidence(lines, spacing, positions, weights, profiles, active):
        if len(positions) == 0:
            return 1., 0., 0.
        def distance(pos):
            upper = np.clip(np.searchsorted(lines, pos), 1, len(lines) - 1)
            return np.minimum(abs(pos - lines[upper]), abs(pos - lines[upper - 1])) / spacing
        d = distance(positions)
        error = float(np.average(np.minimum((d / .22) ** 2, 1.), weights=weights))
        support = float(np.average(d < max(.08, .7 / spacing), weights=weights))
        agreements = []
        for p in profiles[active]:
            peaks, _ = find_peaks(p, prominence=max(.05, float(p.max()) * .12))
            if len(peaks) >= 2:
                agreements.append(float(np.average(distance(peaks) < max(.1, .8 / spacing), weights=p[peaks])))
        return error, support, float(np.mean(agreements)) if agreements else 0.
    ex = actual_evidence(xs, candidate.sx, features.edge_positions_x, features.edge_weights_x,
                         features.profiles_x, features.active_x)
    ey = actual_evidence(ys, candidate.sy, features.edge_positions_y, features.edge_weights_y,
                         features.profiles_y, features.active_y)
    return replace(candidate, x_lines=xs, y_lines=ys, warped=xchanged or ychanged,
                   warp_penalty=(xp + yp) / 2, edge_score=(ex[0] + ey[0]) / 2,
                   support=min(ex[1], ey[1]), strip_agreement=min(ex[2], ey[2]),
                   metadata={**candidate.metadata, "warp_axes": [bool(xchanged), bool(ychanged)],
                             "initial_edge_score": candidate.edge_score})
