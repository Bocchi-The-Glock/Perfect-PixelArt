"""Deterministic, alpha-aware colour recovery over fractional source cells.

RGB distances are Euclidean distances in normalized sRGB (not perceptual Delta E).
All topology checks use the four edge-sharing neighbours; diagonal contacts do
not count as connections. Sampling never performs all-pairs colour distances.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class CellResult:
    rgba: np.ndarray
    confidence: np.ndarray
    alternatives: np.ndarray | None = None
    alternative_support: np.ndarray | None = None
    structure: dict = field(default_factory=dict)
    # Internal bounded evidence: x/y span and standard deviation per cluster.
    evidence: np.ndarray | None = None
    detail_mask: np.ndarray | None = None


def _validate(rgba, x_lines, y_lines, max_samples):
    rgba = np.asarray(rgba, dtype=np.float32)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or min(rgba.shape[:2]) < 1:
        raise ValueError("rgba must have shape (height, width, 4)")
    if not np.isfinite(rgba).all() or rgba.min() < 0 or rgba.max() > 1:
        raise ValueError("rgba must contain finite values in [0, 1]")
    if int(max_samples) != max_samples or max_samples < 16:
        raise ValueError("max_samples must be an integer >= 16")
    axes = []
    for lines, limit in ((x_lines, rgba.shape[1]), (y_lines, rgba.shape[0])):
        lines = np.asarray(lines, dtype=np.float64)
        if (lines.ndim != 1 or len(lines) < 2 or not np.isfinite(lines).all()
                or np.any(np.diff(lines) <= 0)
                or not np.isclose(lines[0], 0) or not np.isclose(lines[-1], limit)
                or np.any(lines[1:-1] <= 0) or np.any(lines[1:-1] >= limit)):
            raise ValueError("cut lines must increase strictly and cover the whole image")
        lines = lines.copy()
        lines[0], lines[-1] = 0., float(limit)
        axes.append(lines)
    return rgba, *axes


def _axis_samples(left, right, limit):
    indices = np.arange(max(0, int(np.floor(left))), min(limit, int(np.ceil(right))))
    overlaps = np.maximum(0, np.minimum(indices + 1, right) - np.maximum(indices, left))
    return indices, overlaps


def sampling_gradient(rgba):
    """Symmetric local edge magnitude, explicitly ignoring hidden RGB."""
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    out = np.zeros(alpha.shape, np.float32)
    for axis in (0, 1):
        a = [slice(None), slice(None)]
        b = a.copy()
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        diff = np.linalg.norm(rgb[a] - rgb[b], axis=-1) * np.minimum(alpha[a], alpha[b])
        diff += np.abs(alpha[a] - alpha[b])
        out[a] = np.maximum(out[a], diff)
        out[b] = np.maximum(out[b], diff)
    return out


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    ordered = weights[order]
    return values[order[np.searchsorted(np.cumsum(ordered), ordered.sum() * .5)]]


def _cluster(rgb, weights, area_weights, positions, alpha):
    """At most three modes from at most max_samples observations."""
    mean = np.average(rgb, axis=0, weights=weights)
    # This fast path is important for already clean pixel art.
    if np.max(np.ptp(rgb, axis=0)) <= .065:
        return [(mean, 1., 1., (1., 1., .29, .29))]

    # A nearly uniform interior with differences confined to the cell rim is
    # usually antialiasing. Avoid iterative clustering for this common case.
    # Interior outliers and narrow through-going lines fail the rim condition,
    # retaining their alternative-colour evidence for the structure pass.
    near = np.linalg.norm(rgb - mean, axis=1) < .07
    rim = np.min(np.minimum(positions, 1 - positions), axis=1) < .19
    if (np.any(near) and area_weights[near].sum() >= .70 * area_weights.sum()
            and np.all(rim[~near])):
        color = np.average(rgb[near], axis=0, weights=weights[near])
        return [(color, float(area_weights[near].sum() / area_weights.sum()),
                 float(weights[near].sum() / weights.sum()), (1., 1., .29, .29))]

    bins = np.minimum((rgb * 12).astype(np.int32), 11)
    ids = bins[:, 0] * 144 + bins[:, 1] * 12 + bins[:, 2]
    unique, inverse = np.unique(ids, return_inverse=True)
    mass = np.bincount(inverse, weights=weights, minlength=len(unique))
    seeds = []
    for index in np.argsort(-mass, kind="stable"):
        chosen = inverse == index
        color = np.average(rgb[chosen], axis=0, weights=weights[chosen])
        if all(np.linalg.norm(color - previous) > .16 for previous in seeds):
            seeds.append(color)
        if len(seeds) == 3:
            break
    centers = np.asarray(seeds)
    for _ in range(4):
        distance = ((rgb[:, None] - centers[None]) ** 2).sum(axis=2)
        labels = distance.argmin(axis=1)
        updated = centers.copy()
        for index in range(len(centers)):
            chosen = labels == index
            if np.any(chosen):
                updated[index] = np.average(rgb[chosen], axis=0, weights=weights[chosen])
        if np.max(np.abs(updated - centers)) < 1e-5:
            centers = updated
            break
        centers = updated

    modes = []
    total_area = area_weights.sum()
    for index, center in enumerate(centers):
        chosen = labels == index
        if not np.any(chosen):
            continue
        residual = np.linalg.norm(rgb[chosen] - center, axis=1)
        # Tukey-like clipping of within-mode outliers, with a finite noise floor.
        cutoff = max(.035, float(np.median(residual)) * 2.5)
        robust = weights[chosen] * np.minimum(1., cutoff / np.maximum(residual, 1e-8))
        color = np.average(rgb[chosen], axis=0, weights=robust)
        support = float(area_weights[chosen].sum() / max(total_area, 1e-12))
        weighted_support = float(weights[chosen].sum() / weights.sum())
        pos = positions[chosen]
        span = np.ptp(pos, axis=0) if len(pos) > 1 else np.zeros(2)
        std = np.sqrt(np.average((pos - np.average(pos, axis=0, weights=area_weights[chosen])) ** 2,
                                 axis=0, weights=area_weights[chosen]))
        modes.append((color, support, weighted_support, (*span, *std)))
    return sorted(modes, key=lambda mode: (-mode[2], -mode[1]))


def _cell_alpha(rgba, x_lines, y_lines):
    """Exact box coverage using the continuous integral of source pixel boxes."""
    alpha = rgba[..., 3]
    shape = (len(y_lines) - 1, len(x_lines) - 1)
    if np.all(alpha == 1):
        return np.ones(shape, np.float32)
    if not np.any(alpha):
        return np.zeros(shape, np.float32)
    integral = np.pad(np.cumsum(np.cumsum(alpha, axis=0, dtype=np.float64), axis=1), ((1, 0), (1, 0)))
    h, w = alpha.shape
    x0, y0 = np.floor(x_lines).astype(int), np.floor(y_lines).astype(int)
    x0, y0 = np.clip(x0, 0, w), np.clip(y0, 0, h)
    x1, y1 = np.minimum(x0 + 1, w), np.minimum(y0 + 1, h)
    tx, ty = x_lines - x0, y_lines - y0
    along0 = integral[y0[:, None], x0] * (1 - tx) + integral[y0[:, None], x1] * tx
    along1 = integral[y1[:, None], x0] * (1 - tx) + integral[y1[:, None], x1] * tx
    values = along0 * (1 - ty[:, None]) + along1 * ty[:, None]
    area = np.diff(y_lines)[:, None] * np.diff(x_lines)[None]
    return np.clip(np.diff(np.diff(values, axis=0), axis=1) / area, 0, 1).astype(np.float32)


def _batched_simple_cells(rgba, x_lines, y_lines, gradient, max_samples,
                          output, confidence, alternatives, supports, evidence):
    """Vectorize the existing single-mode decisions; complex cells fall through.

    Cells sharing integer footprint dimensions use the identical deterministic
    strata as the scalar path, in batches of at most 512 cells and 128 (default)
    samples per cell. Fractional geometry and alpha weights are unchanged.
    This avoids tens of thousands of Python calls on overfine grid candidates.
    """
    ny, nx = output.shape[:2]
    alpha = _cell_alpha(rgba, x_lines, y_lines)
    output[..., 3] = alpha
    resolved = alpha <= 1e-8
    confidence[resolved] = 1
    supports[..., 0][resolved] = 1
    xleft, xright = x_lines[:-1], x_lines[1:]
    yleft, yright = y_lines[:-1], y_lines[1:]
    xstart = np.floor(xleft).astype(int)
    ystart = np.floor(yleft).astype(int)
    widths = np.ceil(xright).astype(int) - xstart
    heights = np.ceil(yright).astype(int) - ystart
    for height in np.unique(heights):
        rows = np.flatnonzero(heights == height)
        for width in np.unique(widths):
            cols = np.flatnonzero(widths == width)
            row_index = np.repeat(rows, len(cols))
            col_index = np.tile(cols, len(rows))
            remaining = ~resolved[row_index, col_index]
            row_index, col_index = row_index[remaining], col_index[remaining]
            sy = min(height, max(1, int(np.sqrt(max_samples * height / width))))
            sx = min(width, max(1, max_samples // sy))
            iy = np.minimum(((np.arange(sy) + .5) * height / sy).astype(int), height - 1)
            ix = np.minimum(((np.arange(sx) + .5) * width / sx).astype(int), width - 1)
            for start in range(0, len(row_index), 512):
                r, c = row_index[start:start + 512], col_index[start:start + 512]
                xi, yi = xstart[c, None] + ix, ystart[r, None] + iy
                pixels = rgba[yi[:, :, None], xi[:, None, :]].reshape(len(r), -1, 4)
                xw = np.maximum(0, np.minimum(xi + 1, xright[c, None]) - np.maximum(xi, xleft[c, None]))
                yw = np.maximum(0, np.minimum(yi + 1, yright[r, None]) - np.maximum(yi, yleft[r, None]))
                area = (yw[:, :, None] * xw[:, None, :]).reshape(len(r), -1) * pixels[..., 3]
                valid = area > 1e-12
                mass = area.sum(axis=1)
                rgb = pixels[..., :3]
                px = np.clip((xi + .5 - xleft[c, None]) / (xright[c] - xleft[c])[:, None], 0, 1)
                py = np.clip((yi + .5 - yleft[r, None]) / (yright[r] - yleft[r])[:, None], 0, 1)
                spatial = .45 + .55 * np.maximum(0, 1 - 2 * ((px[:, None, :] - .5) ** 2 + (py[:, :, None] - .5) ** 2))
                grad = gradient[yi[:, :, None], xi[:, None, :]].reshape(len(r), -1)
                weights = area * spatial.reshape(len(r), -1) / (1 + 4 * grad)
                weight_sum = weights.sum(axis=1)
                mean = (rgb * weights[..., None]).sum(axis=1) / np.maximum(weight_sum[:, None], 1e-30)
                maximum = np.max(np.where(valid[..., None], rgb, -np.inf), axis=1)
                minimum = np.min(np.where(valid[..., None], rgb, np.inf), axis=1)
                pure = (np.max(maximum - minimum, axis=1) <= .065) & (mass > 1e-12)
                near = (np.linalg.norm(rgb - mean[:, None], axis=2) < .07) & valid
                rim = ((np.minimum(px, 1 - px)[:, None, :] < .19)
                       | (np.minimum(py, 1 - py)[:, :, None] < .19)).reshape(len(r), -1)
                near_area = (area * near).sum(axis=1)
                rim_only = ((near_area >= .70 * mass) & np.all(rim | near | ~valid, axis=1)
                            & (mass > 1e-12))
                simple = pure | rim_only
                if not np.any(simple):
                    continue
                color = mean.copy()
                selected = rim_only & ~pure
                near_weight = weights * near
                color[selected] = ((rgb * near_weight[..., None]).sum(axis=1)
                                   / np.maximum(near_weight.sum(axis=1)[:, None], 1e-30))[selected]
                source_support = np.where(pure, 1., near_area / np.maximum(mass, 1e-30))
                residual = (np.minimum(np.linalg.norm(rgb - color[:, None], axis=2), .5) * area).sum(axis=1) / np.maximum(mass, 1e-30)
                rr, cc = r[simple], c[simple]
                output[rr, cc, :3] = color[simple]
                confidence[rr, cc] = np.clip(source_support[simple] * np.exp(-3 * residual[simple]), 0, 1)
                alternatives[rr, cc, 0] = output[rr, cc]
                supports[rr, cc, 0] = source_support[simple]
                evidence[rr, cc, 0] = [1., 1., .29, .29]
                resolved[rr, cc] = True
    return resolved


def recover_cells(rgba, x_lines, y_lines, method="robust", max_samples=128, *, gradient=None):
    """Recover one RGBA pixel per rectangle, including partial image-edge cells.

    Alpha is the exact area-weighted source coverage in robust/median modes.
    Fully transparent pixels contribute no RGB evidence. RGB samples are
    deterministic spatial strata; their count is bounded even for large cells.
    """
    if method not in {"robust", "center", "median"}:
        raise ValueError("sampling must be robust, center or median")
    rgba, x_lines, y_lines = _validate(rgba, x_lines, y_lines, max_samples)
    height, width = rgba.shape[:2]
    ny, nx = len(y_lines) - 1, len(x_lines) - 1
    output = np.zeros((ny, nx, 4), np.float32)
    confidence = np.zeros((ny, nx), np.float32)
    alternatives = np.zeros((ny, nx, 3, 4), np.float32)
    supports = np.zeros((ny, nx, 3), np.float32)
    evidence = np.zeros((ny, nx, 3, 4), np.float32)
    grad = None
    if method == "robust":
        grad = sampling_gradient(rgba) if gradient is None else np.asarray(gradient, dtype=np.float32)
        if grad.shape != rgba.shape[:2] or not np.isfinite(grad).all() or np.any(grad < 0):
            raise ValueError("gradient must be a finite nonnegative H by W array")
    resolved = (_batched_simple_cells(rgba, x_lines, y_lines, grad, max_samples,
                                     output, confidence, alternatives, supports, evidence)
                if method == "robust" else np.zeros((ny, nx), dtype=bool))
    xs = [_axis_samples(a, b, width) for a, b in zip(x_lines[:-1], x_lines[1:])]
    ys = [_axis_samples(a, b, height) for a, b in zip(y_lines[:-1], y_lines[1:])]
    for row, (yi, yw) in enumerate(ys):
        for col, (xi, xw) in enumerate(xs):
            if resolved[row, col]:
                continue
            cx = np.clip(int((x_lines[col] + x_lines[col + 1]) * .5), 0, width - 1)
            cy = np.clip(int((y_lines[row] + y_lines[row + 1]) * .5), 0, height - 1)
            if method == "center":
                output[row, col] = rgba[cy, cx]
                if output[row, col, 3] == 0:
                    output[row, col, :3] = 0
                confidence[row, col] = .5
                alternatives[row, col, 0] = output[row, col]
                supports[row, col, 0] = 1.
                continue
            patch = rgba[yi[0]:yi[-1] + 1, xi[0]:xi[-1] + 1]
            coverage = yw[:, None] * xw[None, :]
            alpha = float((patch[..., 3] * coverage).sum() / coverage.sum())
            output[row, col, 3] = alpha
            if alpha <= 1e-8:
                confidence[row, col] = 1.
                supports[row, col, 0] = 1.
                continue
            # A rectangular stratification avoids biasing samples to rows.
            sy = min(len(yi), max(1, int(np.sqrt(max_samples * len(yi) / len(xi)))))
            sx = min(len(xi), max(1, max_samples // sy))
            iy = np.minimum(((np.arange(sy) + .5) * len(yi) / sy).astype(int), len(yi) - 1)
            ix = np.minimum(((np.arange(sx) + .5) * len(xi) / sx).astype(int), len(xi) - 1)
            samples = patch[iy[:, None], ix].reshape(-1, 4)
            area = coverage[iy[:, None], ix].ravel() * samples[:, 3]
            posx = (xi[ix] + .5 - x_lines[col]) / (x_lines[col + 1] - x_lines[col])
            posy = (yi[iy] + .5 - y_lines[row]) / (y_lines[row + 1] - y_lines[row])
            px, py = np.meshgrid(np.clip(posx, 0, 1), np.clip(posy, 0, 1))
            pos = np.column_stack((px.ravel(), py.ravel()))
            valid = area > 1e-12
            if not np.any(valid):
                # Stratification can miss a tiny visible island. Find the
                # strongest actual sample; never borrow a hidden RGB value.
                strongest = np.argmax(patch[..., 3] * coverage)
                rgb = patch.reshape(-1, 4)[strongest, :3]
                output[row, col, :3] = rgb
                alternatives[row, col, 0] = output[row, col]
                supports[row, col, 0] = 1.
                confidence[row, col] = .2
                continue
            rgb, area, pos = samples[valid, :3], area[valid], pos[valid]
            if method == "median":
                output[row, col, :3] = [_weighted_median(rgb[:, ch], area) for ch in range(3)]
                confidence[row, col] = float(np.exp(-5 * np.average(np.linalg.norm(rgb - output[row, col, :3], axis=1), weights=area)))
                alternatives[row, col, 0] = output[row, col]
                supports[row, col, 0] = 1.
                continue
            local_grad = grad[yi[iy, None], xi[ix]].ravel()[valid]
            # Keep at least 45% spatial weight at corners. The centre alone
            # cannot overrule broad colour support.
            spatial = .45 + .55 * np.maximum(0, 1 - 2 * ((pos - .5) ** 2).sum(axis=1))
            weights = area * spatial / (1 + 4 * local_grad)
            modes = _cluster(rgb, weights, area, pos, alpha)
            output[row, col, :3] = modes[0][0]
            residual = np.average(np.minimum(np.linalg.norm(rgb - modes[0][0], axis=1), .5), weights=area)
            confidence[row, col] = np.clip(modes[0][1] * np.exp(-3 * residual), 0, 1)
            for index, (color, support, _, span) in enumerate(modes):
                alternatives[row, col, index, :3] = color
                alternatives[row, col, index, 3] = alpha
                supports[row, col, index] = support
                evidence[row, col, index] = span
    return CellResult(output, confidence, alternatives, supports,
                      {"sampling": method, "connectivity": 4, "structure_changes": 0,
                       "max_rgb_samples_per_cell": int(max_samples)}, evidence)


def _detail_mask(rgba):
    """Local four-neighbour edge salience and visible dot/hole markers."""
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    strength = np.zeros(alpha.shape, np.float32)
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        distance = np.linalg.norm(rgb[a] - rgb[b], axis=-1) * np.minimum(alpha[a], alpha[b])
        distance += np.abs(alpha[a] - alpha[b])
        strength[a] += distance
        strength[b] += distance
    return strength


def improve_structure(cells, rgba, x_lines, y_lines):
    """One conservative round for source-supported straight lines and highlights.

    No isolated-pixel removal is performed. Dots and holes in the recovered
    representation are recorded/protected for optional palette assignment.
    A line minority requires 8% sampled source coverage, a thin elongated
    source shape, and matching evidence in both four-neighbour cells. An
    isolated highlight requires 6% coverage, a compact multi-sample source
    cluster, strong positive brightness contrast and a uniform neighbourhood.
    """
    output = cells.rgba.copy()
    confidence = cells.confidence.copy()
    changes = 0
    highlight_changes = 0
    support, candidates, evidence = cells.alternative_support, cells.alternatives, cells.evidence
    ny, nx = output.shape[:2]
    if (cells.structure.get("sampling") == "robust" and support is not None
            and candidates is not None and evidence is not None):
        def matching(row, col, color):
            if not (0 <= row < ny and 0 <= col < nx):
                return False
            distance = np.linalg.norm(candidates[row, col, :, :3] - color, axis=1)
            return bool(np.any((distance < .10) & (support[row, col] >= .08)
                               & (candidates[row, col, :, 3] > .25)))
        for row in range(ny):
            for col in range(nx):
                for index in (1, 2):
                    if not (.06 <= support[row, col, index] <= .48):
                        continue
                    color = candidates[row, col, index, :3]
                    if candidates[row, col, index, 3] < .5:
                        continue
                    spanx, spany, stdx, stdy = evidence[row, col, index]
                    enough_line = support[row, col, index] >= .08
                    vertical = enough_line and spany >= .68 and stdx <= .13 and matching(row - 1, col, color) and matching(row + 1, col, color)
                    horizontal = enough_line and spanx >= .68 and stdy <= .13 and matching(row, col - 1, color) and matching(row, col + 1, color)
                    if vertical or horizontal:
                        output[row, col, :3] = color
                        confidence[row, col] = min(float(confidence[row, col]), .65)
                        changes += 1
                        break
                    compact = (.10 <= spanx <= .55 and .10 <= spany <= .55
                               and .035 <= stdx <= .20 and .035 <= stdy <= .20)
                    brighter = float(np.dot(color - cells.rgba[row, col, :3], [.2126, .7152, .0722])) > .30
                    if compact and brighter and 0 < row < ny - 1 and 0 < col < nx - 1:
                        coords = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
                        neighbors = np.asarray([cells.rgba[r, c] for r, c in coords])
                        uniform = np.all(np.linalg.norm(neighbors[:, :3] - cells.rgba[row, col, :3], axis=1) < .10)
                        isolated_source = all(not matching(r, c, color) for r, c in coords)
                        if uniform and np.all(neighbors[:, 3] > .5) and isolated_source:
                            output[row, col, :3] = color
                            confidence[row, col] = min(float(confidence[row, col]), .55)
                            changes += 1
                            highlight_changes += 1
                            break

    salience = _detail_mask(output)
    details = salience > .7
    holes = 0
    isolated = 0
    for row in range(1, ny - 1):
        for col in range(1, nx - 1):
            neighbors = output[[row - 1, row + 1, row, row], [col, col, col - 1, col + 1]]
            value = output[row, col]
            if value[3] < .1 and np.all(neighbors[:, 3] > .8):
                holes += 1
                details[row, col] = True
            if value[3] > .5:
                contrast = np.linalg.norm(neighbors[:, :3] - value[:3], axis=1)
                if np.all(contrast > .18) and np.all(neighbors[:, 3] > .5):
                    isolated += 1
                    details[row, col] = True
    info = dict(cells.structure)
    info.update(structure_changes=changes, supported_holes=holes,
                promoted_highlights=highlight_changes, retained_isolated_details=isolated,
                detail_cells=int(details.sum()), connectivity=4)
    return replace(cells, rgba=output, confidence=confidence, detail_mask=details, structure=info)
