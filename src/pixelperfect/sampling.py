"""Centre-first sampling with a bounded neighbourhood check for isolated impulses."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CellResult:
    rgba: np.ndarray
    confidence: np.ndarray
    structure: dict = field(default_factory=dict)


def _resolve_alpha_mode(alpha, requested):
    if requested not in ("auto", "binary", "coverage"):
        raise ValueError("alpha_mode must be auto, binary, or coverage")
    visible = np.count_nonzero(alpha > 0)
    near_opaque = np.count_nonzero(alpha >= .94) / max(visible, 1)
    return ("sample" if requested == "auto" else requested), float(near_opaque)


def recover_cells(rgba, x_lines, y_lines, method="robust", alpha_mode="auto"):
    if method not in ("robust", "center", "median"):
        raise ValueError("unknown sampling method")
    resolved_alpha, near_opaque = _resolve_alpha_mode(rgba[..., 3], alpha_mode)
    h, w = rgba.shape[:2]
    for lines, length in ((x_lines, w), (y_lines, h)):
        lines = np.asarray(lines)
        if (lines.ndim != 1 or len(lines) < 2 or not np.isfinite(lines).all()
                or lines[0] != 0 or lines[-1] != length or np.any(np.diff(lines) <= 0)):
            raise ValueError("cut lines must be finite, increasing and cover the complete input")
    xs, ys = np.rint(x_lines).astype(int), np.rint(y_lines).astype(int)
    widths, heights = np.diff(xs), np.diff(ys)
    if np.any(widths < 1) or np.any(heights < 1):
        raise ValueError("cut lines must cover at least one source pixel")
    ny, nx = len(heights), len(widths)
    area = heights[:, None] * widths[None]
    alpha = np.add.reduceat(np.add.reduceat(rgba[..., 3], ys[:-1], axis=0), xs[:-1], axis=1) / area
    result = np.zeros((ny * nx, 4), np.float32)
    confidence = np.zeros(ny * nx, np.float32)
    fractions = np.array([.18, .34, .50, .66, .82]) if method != "center" else np.array([.5])
    xp = np.minimum(xs[:-1, None] + (widths[:, None] * fractions).astype(int), xs[1:, None] - 1)
    yp = np.minimum(ys[:-1, None] + (heights[:, None] * fractions).astype(int), ys[1:, None] - 1)
    detail_count = rejected_count = 0
    for start in range(0, ny * nx, 1024):
        ids = np.arange(start, min(start + 1024, ny * nx))
        ids = ids[alpha.ravel()[ids] > 0]
        if not len(ids):
            continue
        yy, xx = ids // nx, ids % nx
        samples = rgba[yp[yy, :, None], xp[xx, None, :]].reshape(len(ids), -1, 4)
        rgb, a = samples[..., :3], samples[..., 3]
        valid = a > 1e-6
        centre = samples[:, len(fractions)**2 // 2].copy()
        centre[centre[:, 3] == 0, :3] = 0
        color, selected_alpha = centre[:, :3].copy(), centre[:, 3].copy()
        support = np.ones(len(ids), np.float32)
        if method != "center":
            count = valid.sum(axis=1)
            sorted_rgb = np.sort(np.where(valid[..., None], rgb, np.inf), axis=1)
            median = sorted_rgb[np.arange(len(ids)), np.maximum(0, (count - 1) // 2)]
            median[count == 0] = 0
            near_median = valid & (np.max(abs(rgb - median[:, None]), axis=2) < .12)
            if method == "median":
                color = median
                selected_alpha = np.median(a, axis=1)
                support = near_median.sum(axis=1) / np.maximum(count, 1)
            else:
                # Compare premultiplied colour AND alpha. Hidden transparent RGB
                # cannot turn a transparent centre into a spurious colour outlier.
                cx, cy = xs[xx] + widths[xx] // 2, ys[yy] + heights[yy] // 2
                px = np.clip(cx[:, None] + [-1, 0, 1], xs[xx, None], xs[xx + 1, None] - 1)
                py = np.clip(cy[:, None] + [-1, 0, 1], ys[yy, None], ys[yy + 1, None] - 1)
                patch = rgba[py[:, :, None], px[:, None, :]].reshape(-1, 9, 4)
                central_pm = centre[:, :3] * centre[:, 3, None]
                distance = np.maximum(np.max(abs(patch[..., :3] * patch[..., 3, None]
                                                - central_pm[:, None]), axis=2),
                                      abs(patch[..., 3] - centre[:, 3, None]))
                near = distance < .10
                # Two adjacent supporters retain a one-source-pixel-wide line;
                # a 2x2 highlight also passes. A single impulse does not.
                coherent = near.sum(axis=1) >= 3
                tiny = (widths[xx] <= 2) | (heights[yy] <= 2)
                deviation = np.max(abs(centre[:, :3] - median), axis=1) > .10
                reject = ~coherent & ~tiny & (deviation | (abs(centre[:, 3] - np.median(a, axis=1)) > .10))
                color[reject] = median[reject]
                selected_alpha[reject] = np.median(a[reject], axis=1)
                rejected_count += int(reject.sum())
                detail_count += int(np.count_nonzero(coherent & deviation & (centre[:, 3] > 0)))
                support = np.where(reject, near_median.sum(axis=1) / np.maximum(count, 1),
                                   near.sum(axis=1) / 9)
                # Deliberately keep the supported centre exactly: averaging the
                # whole colour cluster reintroduces edge blends and blurs lines.
        result[ids, :3], result[ids, 3] = color, selected_alpha
        confidence[ids] = support
    if resolved_alpha == "coverage":
        missing = (alpha.ravel() > 0) & (result[:, 3] == 0)
        result[:, 3] = alpha.ravel()
        # Exact premultiplied fallback only for explicitly requested coverage:
        # a tiny corner fragment must not become a whole opaque output cell.
        if missing.any():
            for c in range(3):
                sums = np.add.reduceat(np.add.reduceat(rgba[..., c] * rgba[..., 3],
                                        ys[:-1], axis=0), xs[:-1], axis=1)
                means = sums / np.maximum(alpha * area, 1e-8)
                result[missing, c] = means.ravel()[missing]
    elif resolved_alpha == "binary":
        result[:, 3] = result[:, 3] >= .5
    result[result[:, 3] <= 0, :3] = 0
    return CellResult(result.reshape(ny, nx, 4), confidence.reshape(ny, nx),
                      dict(supported_central_strokes=detail_count, rejected_central_impulses=rejected_count,
                           samples_per_cell=len(fractions)**2, alpha_mode_requested=alpha_mode,
                           alpha_mode=resolved_alpha, source_near_opaque_fraction=near_opaque,
                           contour_expansion=False))
