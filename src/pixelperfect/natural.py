"""Conservative continuous-image rendering, separate from pixel-grid recovery."""
import numpy as np
from .grid import GridCandidate
from .sampling import recover_cells


def route_image(rgba, features, chosen, config):
    """Prefer recovered/native grids; synthesize a rendering grid only with soft evidence.

    This is an abstaining heuristic, not a calibrated pixel-art/photo classifier.
    A generated grid can never be coarser than an existing accepted grid.
    """
    h, w = rgba.shape[:2]
    report = dict(applied=False, mode=config.photo_mode,
                  curvature_ratio=list(features.ramp_ratio))

    def keep(reason):
        report['reason'] = reason
        return chosen, report

    if config.photo_mode == 'off':
        return keep('ordinary-image rendering disabled')
    if max(w, h) <= 192 or min(w, h) < 16:
        return keep('limited source resolution; preserve existing recovery')
    if chosen is not None:
        source = chosen.metadata['source']
        if chosen.metadata.get('native_preserved') or source in (
                'validated interpolation knots', 'exact two-pixel repetition'):
            return keep('native pixels or validated resampling grid')
        metrics = chosen.metadata.get('axis_metrics', [])
        if len(metrics) == 2:
            unit_support = min(m['unit_gaps'] for m in metrics)
            aligned = min(m['edge_fit'] for m in metrics) > .65 and unit_support > .30
            # Drifting pseudo pixels can have weak global phase alignment while
            # neighbouring edges still repeat the cell interval in both axes.
            # This is a preservation veto, not a claim that the grid is perfect.
            if aligned or unit_support > .45:
                return keep('repeated cell intervals or aligned grid in both axes')
    if min(features.ramp_ratio) >= 1.25:
        return keep('sharp pixel-like transitions; abstain from ordinary-image rendering')

    # Inspect representative original-resolution rows, never a resized grid signal.
    rows = np.unique(np.linspace(0, h - 1, min(h, 64)).astype(int))
    visible = rgba[rows, :, 3] > .05
    samples = rgba[rows, :, :3][visible]
    if len(samples) < 16 or np.max(np.ptp(samples, axis=0)) < .06:
        return keep('flat or insufficient visible colour evidence')

    # About 4 source pixels per rendered pixel, bounded to 96..256 on the long side.
    # Sparse transparent subjects receive a finer budget to avoid losing their detail.
    target = min(256, max(96, round(max(w, h) / 4)))
    spacing = max(w, h) / target
    alpha = rgba[..., 3]
    if np.any(alpha <= .01):
        yy = np.flatnonzero(np.any(alpha > .05, axis=1))
        xx = np.flatnonzero(np.any(alpha > .05, axis=0))
        if len(xx) and len(yy):
            spacing = min(spacing, max(1., max(xx[-1] - xx[0] + 1, yy[-1] - yy[0] + 1) / 128))
    if spacing < 1.5:
        return keep('small visible subject; avoid further reduction')
    nx, ny = max(1, round(w / spacing)), max(1, round(h / spacing))
    if chosen is not None and (nx < len(chosen.x_lines) - 1 or ny < len(chosen.y_lines) - 1):
        return keep('existing grid retains more detail than the rendering budget')

    if config.square:
        # Equal nominal step; partial edge cells still cover the entire input.
        xs = np.r_[np.arange(0., w - .5, spacing), float(w)]
        ys = np.r_[np.arange(0., h - .5, spacing), float(h)]
        sx = sy = spacing
    else:
        xs, ys = np.linspace(0, w, nx + 1), np.linspace(0, h, ny + 1)
        sx, sy = w / nx, h / ny
    # Verify the final counts too, including partial cells in square mode.
    if chosen is not None and (len(xs) < len(chosen.x_lines) or len(ys) < len(chosen.y_lines)):
        return keep('generated grid would lose recovered cells')
    report.update(applied=True, reason='soft image without convincing pixel-grid evidence',
                  generated_size=[len(xs) - 1, len(ys) - 1],
                  previous_grid=None if chosen is None else dict(
                      size=[len(chosen.x_lines) - 1, len(chosen.y_lines) - 1],
                      spacing=[chosen.sx, chosen.sy], score=chosen.support),
                  confidence_note='generated rendering grid, not a recovered source lattice')
    return GridCandidate(sx, sy, 0., 0., xs, ys, 0., False,
                         {'source': 'ordinary image rendering', 'stylized': True}), report


def render_cells(rgba, grid, config):
    """Average smooth regions; retain centre-supported colours at strong boundaries.

    Statistics use premultiplied colour. Default alpha remains sampled, so an
    intersecting opaque shape never fills an otherwise transparent whole cell.
    The original recovery sampler is not modified.
    """
    cells = recover_cells(rgba, grid.x_lines, grid.y_lines, config.sampling,
                          alpha_mode=config.alpha_mode)
    cells.structure['rendering'] = 'ordinary image'
    if config.sampling != 'robust':
        return cells
    xs, ys = np.rint(grid.x_lines).astype(int), np.rint(grid.y_lines).astype(int)

    def sums(values):
        return np.add.reduceat(np.add.reduceat(values, ys[:-1], axis=0), xs[:-1], axis=1)

    alpha = rgba[..., 3]
    mass = sums(alpha)
    means = np.zeros_like(cells.rgba[..., :3])
    variance = np.zeros_like(mass)
    for channel in range(3):
        values = rgba[..., channel]
        mean = sums(values * alpha) / np.maximum(mass, 1e-8)
        variance = np.maximum(variance, sums(values * values * alpha) / np.maximum(mass, 1e-8) - mean * mean)
        means[..., channel] = mean
    smooth = (variance < .06 ** 2) & (cells.rgba[..., 3] > 0)
    cells.rgba[smooth, :3] = means[smooth]
    cells.rgba[cells.rgba[..., 3] == 0, :3] = 0
    cells.structure.update(rendering_sampler='area in smooth regions; supported centre at boundaries',
                           averaged_smooth_cells=int(smooth.sum()))
    return cells
