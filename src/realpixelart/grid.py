"""Small Fourier/gap proposal set, cheap edge validation, then one lattice."""
from dataclasses import dataclass, field, replace
import numpy as np
from .features import peaks, smooth, axis_segment_evidence

_MIN_GRID_SCORE = .30


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


def make_lines(length, spacing, phase=0.):
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
    report = {"axis_proposals": [a["proposals"] for a in axes], "candidates": [],
              "axis_evidence": [{"peaks": len(a["pos"]), "contrast": a["contrast"]} for a in axes]}
    if any(len(a["pos"]) < 4 or a["contrast"] < .25 for a in axes):
        report["rejection"] = "insufficient edge peaks or projection contrast"
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
        report["rejection"] = "no candidate spacing within search range"
        return None, report
    finalists.sort(key=lambda c: (-c[0], -c[1][0]))
    # FFT bins/gap modes propose a scale, not a continuous optimum. With drifting
    # contours a small spacing change also changes which local edges _walk uses.
    # Before rejecting a near-threshold result, examine a bounded neighbourhood
    # of the three finalists. Successful existing detections remain untouched.
    initial_score = finalists[0][0]
    report["spacing_refinement"] = {"attempted": False, "initial_score": initial_score,
                                    "candidates": []}
    if (.25 <= initial_score < _MIN_GRID_SCORE
            and min(m["unit_gaps"] for m in finalists[0][5]) >= .10):
        extra = []
        seen = [ss for _, ss, _, _, _, _ in finalists]
        for _, sizes, origin, _, _, _ in finalists:
            # At most five trials per finalist. Scale both axes together to keep
            # the original aspect ratio and the square-mode constraint.
            step = max(.25, round(min(sizes) * .02 * 4) / 4)
            centre = round(sizes[0] / step) * step
            for offset in (-2, -1, 0, 1, 2):
                sx = centre + offset * step
                ss = (sx, sizes[1] * sx / sizes[0])
                if not (config.min_pixel_size <= min(ss) and
                        max(ss) <= min(config.max_pixel_size, w / 2, h / 2)):
                    continue
                if any(np.allclose(ss, old, rtol=0., atol=1e-7) for old in seen):
                    continue
                seen.append(ss)
                pp = [_phase(a, s) for a, s in zip(axes, ss)]
                cc = [_walk(a, s, p, config.local_warp == "auto") for a, s, p in zip(axes, ss, pp)]
                mm = [_measure(a, s, c) for a, s, c in zip(axes, ss, cc)]
                value = float(np.mean([m["score"] for m in mm]))
                extra.append((value, ss, origin + " local spacing refinement", pp, cc, mm))
        report["spacing_refinement"].update(attempted=True, candidates=[
            {"spacing": list(ss), "score": value, "axes": mm} for value, ss, _, _, _, mm in extra])
        # The same acceptance thresholds apply; adding trials is not permission
        # to lower the evidence requirement or report an artificial confidence.
        finalists.extend(c for c in extra if min(m["unit_gaps"] for m in c[5]) >= .10)
        finalists.sort(key=lambda c: (-c[0], -c[1][0]))
    score, sizes, origin, phases, lines, metrics = finalists[0]
    report["candidates"] = [{"spacing": list(ss), "source": src, "score": value, "axes": mm}
                            for value, ss, src, _, _, mm in ranked]
    report["refined"] = [{"spacing": list(ss), "score": value, "axes": mm}
                         for value, ss, _, _, _, mm in finalists]
    report["selected_score"] = score
    if score < _MIN_GRID_SCORE or min(m["unit_gaps"] for m in metrics) < .10:
        report["rejection"] = (f"grid score below {_MIN_GRID_SCORE:.2f}" if score < _MIN_GRID_SCORE else
                               "unit cell interval support below 0.10 in one axis")
        return None, report
    warped = any(len(c) != len(make_lines(n, s, p)) or not np.allclose(c, make_lines(n, s, p))
                 for c, n, s, p in zip(lines, (w, h), sizes, phases))
    return GridCandidate(*sizes, *phases, *lines, float(np.clip(score, 0., 1.)), warped,
                         {"source": origin, "axis_metrics": metrics}), report


def _coarse_grid(features, shape, config):
    edge, report = _detect_grid(features, shape, config)
    report["ramp_curvature_ratio"] = list(features.ramp_ratio)
    report["evidence_model"] = "colour boundaries"
    if max(features.ramp_ratio) >= .65:
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


def _native_resolution(features, shape, chosen, report):
    """Conservative one-pixel alternative, independent of filename and image size.

    A perfect per-pixel reconstruction error alone would make every photograph
    win. Require repeated sharp one-pixel details in both axes AND weak alignment
    of raw edges to the proposed coarser grid instead.
    """
    axes = features.native_axes
    boundary_fit = []
    if chosen is not None:
        for g, cuts, direction in ((features.gradient_x, chosen.x_lines, 0),
                                   (features.gradient_y, chosen.y_lines, 1)):
            mass = np.minimum(g, .35) * (g > .06)
            profile = mass.sum(axis=direction)
            positions = np.unique(np.clip(np.rint(cuts).astype(int), 0, len(profile) - 1))
            boundary_fit.append(float(profile[positions].sum() / max(float(profile.sum()), 1e-9)))
    sharp = min(features.ramp_ratio) >= 1.6
    detail = all(a['turn_fraction'] >= .08 and a['turn_count'] >= 8
                 and a['supporting_lines'] >= 3 for a in axes)
    coarse_supported = len(boundary_fit) == 2 and min(boundary_fit) >= .80
    # Without a competing grid, sharp reversals alone also describe white or
    # grain noise. Require repeated flat pixel regions alongside those details.
    flat_fraction = [float(np.count_nonzero(g < .005) / g.size)
                     for g in (features.gradient_x, features.gradient_y)]
    preserve = sharp and detail and not coarse_supported and (chosen is not None or min(flat_fraction) > .20)
    report['native_resolution'] = dict(
        axes=list(axes), sharpness_ratio=list(features.ramp_ratio),
        coarse_boundary_fit=boundary_fit, selected=bool(preserve),
        flat_neighbor_fraction=flat_fraction,
        reason=('repeated sharp one-pixel detail' if preserve and chosen is None else
                'one-pixel detail contradicts coarse grid' if preserve else 'insufficient native detail evidence'))
    if not preserve:
        return chosen, report
    report['rejected_coarse_grid'] = None if chosen is None else dict(
        spacing=[chosen.sx, chosen.sy], score=chosen.support, source=chosen.metadata['source'])
    report['evidence_model'] = 'native pixel detail'
    # Preservation does not establish a unique original grid.
    confidence = float(min(.75, .5 + min(a['turn_fraction'] for a in axes)))
    report['selected_score'] = confidence
    h, w = shape[:2]
    return GridCandidate(1., 1., 0., 0., np.arange(w + 1, dtype=float),
                         np.arange(h + 1, dtype=float), confidence,
                         metadata={'source': 'native pixel detail', 'native_preserved': True}), report


def _exact_two_pixel_grid(features, shape, config):
    """Recover clean 2x repetition before smoothed projections erase its period.

    All examined changes, including weak changes, must align. Large-image mode
    examines full-length scanlines and records that restricted evidence scope.
    """
    if not config.min_pixel_size <= 2 <= config.max_pixel_size:
        return None
    phases = []
    for g, direction in ((features.gradient_x, 0), (features.gradient_y, 1)):
        pos = np.flatnonzero(g.max(axis=direction) > 1e-6)
        if len(pos) < 4 or np.gcd.reduce(np.diff(pos)) != 2:
            return None
        phases.append(float(pos[0] % 2))
    h, w = shape[:2]
    return GridCandidate(2., 2., *phases, make_lines(w, 2, phases[0]), make_lines(h, 2, phases[1]),
                         .95 if features.mode == 'full image' else .85,
                         metadata={'source': 'exact two-pixel repetition', 'evidence_scope': features.mode})


def _exact_integer_grid(features, shape, config):
    """Rescue rejected integer repeats from unsmoothed edge coordinates.

    Projection peaks can hide adjacent cell boundaries behind stronger repeated
    shapes. A common divisor is usable only when every examined nonzero change
    fits it in both axes; even faint interpolation/noise changes invalidate it.
    """
    sizes, phases = [], []
    for gradient, axis in ((features.gradient_x, 0), (features.gradient_y, 1)):
        positions = np.flatnonzero(gradient.max(axis=axis) > 1e-6)
        if len(positions) < 4:
            return None
        spacing = int(np.gcd.reduce(np.diff(positions)))
        if spacing < max(2, config.min_pixel_size) or spacing > config.max_pixel_size:
            return None
        sizes.append(float(spacing))
        phases.append(float(positions[0] % spacing))
    if (config.square and sizes[0] != sizes[1]) or max(sizes) / min(sizes) > 1.12:
        return None
    h, w = shape[:2]
    return GridCandidate(*sizes, *phases, make_lines(w, sizes[0], phases[0]),
                         make_lines(h, sizes[1], phases[1]),
                         .95 if features.mode == 'full image' else .85,
                         metadata={'source': 'validated integer repetition', 'evidence_scope': features.mode})


def detect_grid(features, shape, config):
    chosen, report = _coarse_grid(features, shape, config)
    exact = _exact_two_pixel_grid(features, shape, config)
    if exact is not None:
        chosen = exact
        report['evidence_model'] = 'exact two-pixel repetition'
        report['exact_two_pixel_grid'] = dict(spacing=[2., 2.], phase=[exact.phase_x, exact.phase_y],
                                             score=exact.support, all_changes_aligned=features.mode == 'full image',
                                             evidence_scope=features.mode)
        report['selected_score'] = exact.support
    elif chosen is None:
        integer = _exact_integer_grid(features, shape, config)
        if integer is not None:
            chosen = integer
            report['evidence_model'] = 'validated integer repetition'
            report['integer_repetition'] = dict(spacing=[integer.sx, integer.sy],
                                                phase=[integer.phase_x, integer.phase_y],
                                                evidence_scope=features.mode)
            report['selected_score'] = integer.support
    return _native_resolution(features, shape, chosen, report)


def validate_grid_segments(rgba, features, chosen, report):
    """Let distributed non-axis-aligned contours veto an already weak lattice.

    Strong grids and native/resampling evidence take precedence. Too few edges
    abstain; a positive segment result never creates a grid or inflates its score.
    """
    evidence = dict(checked=False, decision='unchanged')
    report['axis_segments'] = evidence
    if chosen is None:
        evidence['reason'] = 'no candidate grid to validate'
        return chosen
    if (chosen.metadata.get('native_preserved') or chosen.metadata['source'] in (
            'validated interpolation knots', 'exact two-pixel repetition', 'validated integer repetition')
            or min(chosen.sx, chosen.sy) < 3):
        evidence['reason'] = 'native pixels or validated resampling grid take precedence'
        return chosen
    metrics = chosen.metadata.get('axis_metrics', [])
    repeated = len(metrics) == 2 and min(m['unit_gaps'] for m in metrics) > .45
    aligned = (len(metrics) == 2 and min(m['edge_fit'] for m in metrics) > .65
               and min(m['unit_gaps'] for m in metrics) > .30)
    if chosen.support >= .45 or repeated or aligned:
        evidence['reason'] = 'strong existing grid evidence takes precedence'
        return chosen
    if min(features.ramp_ratio) < 1.25:
        evidence['reason'] = 'soft boundaries; missing straight runs are inconclusive'
        return chosen
    evidence.update(axis_segment_evidence(rgba, (chosen.sx, chosen.sy)), checked=True)
    active = [p for p in evidence['patches'] if p['edges'] >= 32 and p['mass'] >= 1.]
    weak = [p for p in active if p['vertical'] + p['horizontal'] < .25]
    evidence['contradicting_patches'] = len(weak)
    # Requiring both orientation and sustained runs avoids treating short blurred
    # pixel steps as curves. Uniform patch votes keep one long outline from voting
    # for an entire image; blank/transparent patches never count against a grid.
    contradiction = (len(active) >= 4 and len(weak) / len(active) >= .75
                     and evidence['segment_fraction'] < .25 and evidence['axis_fraction'] < .75)
    if not contradiction:
        supported = (len(active) >= 3 and evidence['segment_fraction'] >= .35
                     and evidence['axis_fraction'] >= .75)
        evidence.update(decision='supported' if supported else 'inconclusive',
                        reason='axis-aligned runs support the candidate' if supported else
                               'insufficient contradictory evidence; preserve candidate')
        return chosen
    evidence.update(decision='rejected', reason='weak grid contradicted by non-axis-aligned contours in multiple patches')
    report['segment_rejected_grid'] = dict(spacing=[chosen.sx, chosen.sy], score=chosen.support,
                                           size=[len(chosen.x_lines)-1, len(chosen.y_lines)-1],
                                           source=chosen.metadata['source'])
    report['selected_score'] = 0.
    return None
