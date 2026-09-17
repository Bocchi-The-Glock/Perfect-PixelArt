"""Small Fourier/gap proposal set, cheap edge validation, then one lattice."""
from dataclasses import dataclass, field, replace
import numpy as np
from .features import peaks, smooth


@dataclass
class GridCandidate:
    sx: float
    sy: float
    phase_x: float
    phase_y: float
    x_lines: np.ndarray
    y_lines: np.ndarray
    support: float = 0.
    warped: bool = False
    metadata: dict = field(default_factory=dict)


def make_lines(length, spacing, phase=0., count=None):
    if count is not None:
        return np.linspace(0, length, count + 1)
    inside = np.arange(phase - spacing, length + spacing, spacing)
    inside = inside[(inside >= max(.5, .2 * spacing)) & (inside <= length - max(.5, .2 * spacing))]
    return np.r_[0., inside, float(length)]


def _axis(profile, spectrum, minimum, maximum):
    pos = peaks(profile, max(float(profile.max()) * .16, .0008))
    # Quantized bilinear ramps contain several almost equal maxima. Treat a
    # plateau with no meaningful valley as ONE edge, located at its midpoint.
    # Otherwise the tiny ripple gaps wrongly vote for half-sized cells.
    groups = []
    for p in pos:
        if groups and np.min(profile[groups[-1][-1]:p + 1]) >= .90 * min(profile[groups[-1][-1]], profile[p]):
            groups[-1].append(p)
        else:
            groups.append([p])
    pos = np.array([(g[0] + g[-1]) / 2 for g in groups], dtype=float)
    weights = np.interp(pos, np.arange(len(profile)), profile)
    gaps = np.diff(pos).astype(float)
    gap_weights = np.minimum(weights[:-1], weights[1:])
    proposals = []
    if len(gaps) and maximum >= minimum:
        steps = np.arange(minimum, maximum + .125, .25)
        density = np.array([np.sum(gap_weights * np.exp(-.5 * ((gaps - s) / max(.65, .09 * s)) ** 2))
                            for s in steps])
        ix = peaks(np.r_[-1., density, -1.]) - 1
        for k in ix[np.argsort(density[ix], kind="stable")[-3:][::-1]]:
            proposals.append((float(steps[k]), "edge gaps"))
    power = np.abs(np.fft.rfft(profile - profile.mean()))
    freq = peaks(power)
    freq = freq[(freq > 0) & (len(profile) / freq >= minimum) & (len(profile) / freq <= maximum)]
    for k in freq[np.argsort(power[freq], kind="stable")[-3:][::-1]]:
        proposals.append((len(profile) / float(k), "edge FFT"))
    # Repeated block shapes have spectral troughs near reciprocal block widths.
    log_profile = smooth(smooth(spectrum))
    trough = smooth(log_profile, 12) - log_profile
    freq = peaks(trough)
    freq = freq[(freq > 0) & (len(profile) / freq >= minimum) & (len(profile) / freq <= maximum)]
    for k in freq[np.argsort(trough[freq], kind="stable")[-4:][::-1]]:
        proposals.append((len(profile) / float(k), "image FFT trough"))
    return dict(profile=profile, pos=pos, weights=weights, gaps=gaps, gap_weights=gap_weights,
                power=power, trough=trough, proposals=proposals, contrast=float((np.percentile(profile, 95) - np.percentile(profile, 20)) / max(np.percentile(profile, 95), 1e-9)))


def _phase(axis, spacing):
    pos, weights = axis["pos"], axis["weights"]
    if not len(pos):
        return 0.
    phases = np.linspace(0, spacing, min(64, max(12, int(spacing * 4))), endpoint=False)
    distance = np.abs((pos[None] - phases[:, None] + spacing / 2) % spacing - spacing / 2)
    fit = np.sum(weights * np.exp(-.5 * (distance / max(.6, .12 * spacing)) ** 2), axis=1)
    best = phases[int(np.argmax(fit))]
    for _ in range(2):
        residual = (pos - best + spacing / 2) % spacing - spacing / 2
        keep = np.abs(residual) < max(.8, .2 * spacing)
        if keep.any():
            best = (best + np.average(residual[keep], weights=weights[keep])) % spacing
    return 0. if min(best, spacing - best) < 1e-7 else float(best)


def _walk(axis, spacing, phase, allow_warp):
    length = len(axis["profile"])
    regular = make_lines(length, spacing, phase)
    if not allow_warp or spacing < 3 or len(axis["pos"]) < 4:
        return regular
    pos, weights = axis["pos"], axis["weights"]
    near = np.flatnonzero(np.abs(pos - length / 2) <= spacing)
    if not len(near):
        return regular
    anchor = float(pos[near[np.argmax(weights[near])]])
    cuts = [anchor]
    for direction in (-1, 1):
        current = anchor
        while True:
            predicted = current + direction * spacing
            if predicted <= .2 * spacing or predicted >= length - .2 * spacing:
                break
            lo, hi = np.searchsorted(pos, [predicted - .24 * spacing, predicted + .24 * spacing])
            if hi > lo:
                local = np.arange(lo, hi)
                score = weights[local] * np.exp(-.5 * ((pos[local] - predicted) / (.2 * spacing)) ** 2)
                current = float(pos[local[int(np.argmax(score))]])
            else:
                current = predicted
            cuts.append(current)
    return np.r_[0., sorted(cuts), float(length)]


def _measure(axis, spacing, lines):
    pos, weights = axis["pos"], axis["weights"]
    if len(pos) < 4:
        return dict(score=0., edge_fit=0., unit_gaps=0., fft=0.)
    upper = np.clip(np.searchsorted(lines, pos), 1, len(lines) - 1)
    distance = np.minimum(abs(pos - lines[upper]), abs(pos - lines[upper - 1]))
    tolerance = max(.65, .14 * spacing)
    explained = float(np.average(np.exp(-.5 * (distance / tolerance) ** 2), weights=weights))
    chance = min(.85, 2.5066 * tolerance / spacing)
    explained = max(0., (explained - chance) / (1 - chance))
    gaps, gw = axis["gaps"], axis["gap_weights"]
    units = float(np.average(np.exp(-.5 * ((gaps - spacing) / max(.7, .12 * spacing)) ** 2), weights=gw))
    frequency = len(axis["profile"]) / spacing
    power = axis["power"]
    periodic = float(np.interp(frequency, np.arange(len(power)), power) / max(power[1:].max(initial=0), 1e-9))
    return dict(score=.48 * units + .42 * explained + .10 * periodic,
                edge_fit=explained, unit_gaps=units, fft=periodic)


def _detect_grid(features, shape, config):
    h, w = shape[:2]
    axes = [_axis(p, spec, config.min_pixel_size, min(config.max_pixel_size, n / 2))
            for p, spec, n in [(features.profile_x, features.spectral_x, w),
                               (features.profile_y, features.spectral_y, h)]]
    report = {"axis_proposals": [a["proposals"] for a in axes], "candidates": []}
    if config.target_size is not None:
        tw, th = config.target_size
        return GridCandidate(w / tw, h / th, 0., 0., make_lines(w, w / tw, count=tw),
                             make_lines(h, h / th, count=th), metadata={"source": "target size"}), report
    manual = config.pixel_size is not None
    if manual:
        candidates = [(config.pixel_size, "pixel size")]
    else:
        if any(len(a["pos"]) < 4 or a["contrast"] < .25 for a in axes):
            return None, report
        pool = []
        for a in axes:
            for s, origin in a["proposals"]:
                for factor, label in [(1., ""), (.5, " half"), (2., " double")]:
                    value = s * factor
                    if config.min_pixel_size <= value <= min(config.max_pixel_size, w / 2, h / 2):
                        if not any(abs(value - old[0]) < .08 for old in pool):
                            pool.append((value, origin + label))
        candidates = [((s, s), origin) for s, origin in pool]
        # Accept mildly rectangular spacings only with strong regular evidence
        # in BOTH directions; irregular AI contours keep a common spacing.
        if not config.square:
            best_axes = []
            for axis in axes:
                fits = []
                for spacing, _ in axis["proposals"]:
                    phase = _phase(axis, spacing)
                    pos, weights = axis["pos"], axis["weights"]
                    index = np.rint((pos - phase) / spacing)
                    residual = pos - (phase + index * spacing)
                    keep = abs(residual) <= max(.7, .18 * spacing)
                    if np.count_nonzero(keep) >= 4 and np.ptp(index[keep]) > 0:
                        xx, yy, ww = index[keep], pos[keep], weights[keep]
                        xx = xx - np.average(xx, weights=ww)
                        slope = np.sum(ww * xx * yy) / np.sum(ww * xx * xx)
                        if abs(slope / spacing - 1) < .025:
                            spacing = float(slope)
                    phase = _phase(axis, spacing)
                    metric = _measure(axis, spacing, make_lines(len(axis["profile"]), spacing, phase))
                    fits.append((metric["score"], spacing, metric["edge_fit"]))
                best_axes.append(max(fits, default=(0, 1, 0)))
            bx, by = best_axes
            if min(bx[2], by[2]) > .85 and max(bx[1], by[1]) / min(bx[1], by[1]) <= 1.12:
                if config.min_pixel_size <= min(bx[1],by[1]) and max(bx[1],by[1]) <= config.max_pixel_size:
                    candidates.append(((bx[1], by[1]), "strong regular colour edges"))

    ranked = []
    for sizes, origin in candidates:
        phases, lines, metrics = [], [], []
        for axis, s in zip(axes, sizes):
            phase = _phase(axis, s)
            cuts = make_lines(len(axis["profile"]), s, phase)
            metrics.append(_measure(axis, s, cuts))
            phases.append(phase); lines.append(cuts)
        score = float(np.mean([m["score"] for m in metrics]))
        ranked.append((score, sizes, origin, phases, lines, metrics))
    ranked.sort(key=lambda c: (-c[0], -c[1][0]))
    finalists = []
    for _, sizes, origin, phases, lines, metrics in ranked[:3]:
        refined = [cuts if m["edge_fit"] > .94 else _walk(a, s, phase, config.local_warp == "auto")
                   for a, s, phase, cuts, m in zip(axes, sizes, phases, lines, metrics)]
        ms = [_measure(a, s, cuts) for a, s, cuts in zip(axes, sizes, refined)]
        score = float(np.mean([m["score"] for m in ms]))
        finalists.append((score, sizes, origin, phases, refined, ms))
    if not finalists:
        return None, report
    finalists.sort(key=lambda c: (-c[0], -c[1][0]))
    score, sizes, origin, phases, lines, metrics = finalists[0]
    report["candidates"] = [{"spacing": list(ss), "source": src, "score": value, "axes": mm}
                            for value, ss, src, _, _, mm in ranked]
    report["refined"] = [{"spacing": list(ss), "score": value, "axes": mm}
                         for value, ss, _, _, _, mm in finalists]
    report["selected_score"] = score
    if not manual and (score < .31 or min(m["unit_gaps"] for m in metrics) < .10):
        return None, report
    warped = any(len(c) != len(make_lines(n, s, p)) or not np.allclose(c, make_lines(n, s, p))
                 for c, n, s, p in zip(lines, (w, h), sizes, phases))
    return GridCandidate(*sizes, *phases, *lines, float(np.clip(score, 0., 1.)), warped,
                         {"source": origin, "axis_metrics": metrics}), report


def detect_grid(features, shape, config):
    edge, report = _detect_grid(features, shape, config)
    report["ramp_curvature_ratio"] = list(features.ramp_ratio)
    report["evidence_model"] = "colour boundaries"
    if config.target_size is not None or max(features.ramp_ratio) >= .65:
        return edge, report
    # A separate, strictly gated observation model: linear interpolation has
    # curvature at source pixel CENTRES, not at cell boundaries. FFT remains
    # only a proposal source; measured curvature must support the lattice.
    linear = replace(features, profile_x=features.curvature_x, profile_y=features.curvature_y)
    knot, alternate = _detect_grid(linear, shape, replace(config, local_warp="off"))
    report["interpolation_search"] = alternate
    if knot is None or knot.support < .65:
        return edge, report
    if min(m["edge_fit"] for m in knot.metadata["axis_metrics"]) < .75:
        return edge, report
    if edge is not None and knot.support < edge.support + .08:
        return edge, report
    knot.phase_x = (knot.phase_x + .5 - knot.sx / 2) % knot.sx
    knot.phase_y = (knot.phase_y + .5 - knot.sy / 2) % knot.sy
    knot.x_lines = make_lines(shape[1], knot.sx, knot.phase_x)
    knot.y_lines = make_lines(shape[0], knot.sy, knot.phase_y)
    knot.metadata["source"] = "validated interpolation knots"
    report["evidence_model"] = "linear interpolation knots"
    report["selected_score"] = knot.support
    return knot, report
