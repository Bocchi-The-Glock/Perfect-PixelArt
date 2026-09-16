"""Area-aware forward rendering and comparable, explicitly heuristic costs."""
import numpy as np
from scipy import sparse, ndimage


def _coverage(lines, length):
    """Sparse exact pixel-box / cell overlap, not rounded nearest indexing."""
    rows, cols, weights = [], [], []
    for cell, (lo, hi) in enumerate(zip(lines[:-1], lines[1:])):
        indices = np.arange(max(0, int(np.floor(lo))), min(length, int(np.ceil(hi))))
        overlap = np.maximum(0, np.minimum(indices + 1, hi) - np.maximum(indices, lo))
        rows.extend(indices.tolist())
        cols.extend([cell] * len(indices))
        weights.extend(overlap.tolist())
    return sparse.csr_matrix((weights, (rows, cols)), shape=(length, len(lines) - 1), dtype=np.float32)


def premultiply(rgba):
    result = np.empty_like(rgba, dtype=np.float32)
    result[..., :3] = rgba[..., :3] * rgba[..., 3:4]
    result[..., 3] = rgba[..., 3]
    return result


def render_grid(rgba, x_lines, y_lines, shape_hw):
    """Render cells as exact visible area averages, returning premultiplied RGBA."""
    h, w = shape_hw
    p = premultiply(rgba)
    ax, ay = _coverage(x_lines, w), _coverage(y_lines, h)
    nx, ny = p.shape[1], p.shape[0]
    along_x = (ax @ p.transpose(1, 0, 2).reshape(nx, -1)).reshape(w, ny, 4).transpose(1, 0, 2)
    return np.asarray(ay @ along_x.reshape(ny, -1)).reshape(h, w, 4)


def color_edges(premul):
    """Directional edge norms across black and white composites include alpha."""
    # Both backgrounds make a colored transparent boundary observable and make
    # arbitrary hidden RGB mathematically irrelevant.
    black = premul[..., :3]
    white = black + 1 - premul[..., 3:4]
    output = []
    for axis in (1, 0):
        a = np.diff(black, axis=axis)
        b = np.diff(white, axis=axis)
        e = np.sqrt((np.sum(a * a, axis=2) + np.sum(b * b, axis=2)) / 6)
        output.append(e.astype(np.float32))
    return output


def prepare_evidence(rgba):
    p = premultiply(rgba)
    edges = color_edges(p)
    # Stronger-than-noise evidence. Do not introduce threshold ties as edges.
    strong = []
    for e in edges:
        if e.size:
            threshold = max(0.035, float(e.max()) * 0.12)
            strong.append(np.where(e >= threshold, np.minimum(e, 0.5), 0))
        else:
            strong.append(e)
    return {"premultiplied": p, "edges": strong}


def _robust_error(source, rendered):
    alpha_error = np.abs(source[..., 3] - rendered[..., 3])
    black = np.abs(source[..., :3] - rendered[..., :3])
    white = np.abs((source[..., :3] - source[..., 3:4]) - (rendered[..., :3] - rendered[..., 3:4]))
    def huber(d):
        return np.where(d < 0.08, d * d / 0.16, d - 0.04)
    # Common denominator: every source pixel's area, for all grid dimensions.
    return float(0.75 * np.mean(huber((black + white) * 0.5)) + 0.25 * np.mean(huber(alpha_error)))


def structure_error(evidence, reconstructed):
    """Input-supported edge recall, with 1-source-pixel localization tolerance.

    This is a structure proxy, NOT a semantic or exact topology guarantee.
    Cell-level 4-neighbor line/hole/point rules are implemented in sampling.
    """
    produced = color_edges(reconstructed)
    missing, total = 0.0, 0.0
    for observed, estimated in zip(evidence["edges"], produced):
        if not observed.size:
            continue
        nearby = ndimage.maximum_filter(estimated, size=3, mode="nearest")
        missing += float(np.maximum(observed - nearby, 0).sum(dtype=np.float64))
        total += float(observed.sum(dtype=np.float64))
    return missing / total if total > 1e-8 else 0.0


def expression_complexity(rgba, source_area):
    """An engineering proxy using cells, colors, and spatial transitions.

    All costs are normalized by the SAME source area. No PNG file size enters.
    Quantized bins stabilize the color count under small numerical differences.
    """
    visible = rgba[..., 3] > 0.01
    n = rgba.shape[0] * rgba.shape[1]
    bins = np.clip(np.floor(rgba[..., :3] * 31 + 0.5), 0, 31).astype(np.int16)
    keys = bins[..., 0] * 1024 + bins[..., 1] * 32 + bins[..., 2]
    unique = len(np.unique(keys[visible])) if visible.any() else 0
    transitions = 0
    for axis in (0, 1):
        p = np.diff(premultiply(rgba), axis=axis)
        transitions += int(np.count_nonzero(np.linalg.norm(p, axis=2) > 0.08))
    cell_cost = n / source_area
    color_cost = np.count_nonzero(visible) * np.log2(unique + 1) / (24 * source_area)
    spatial_cost = transitions / (2 * source_area)
    return float(cell_cost + color_cost + spatial_cost), {
        "cells_per_source_pixel": float(cell_cost), "color_expression": float(color_cost),
        "spatial_transitions": float(spatial_cost), "visible_color_bins": unique,
    }


def score_candidate(evidence, cells, grid, config):
    h, w = evidence["premultiplied"].shape[:2]
    rendered = render_grid(cells.rgba, grid.x_lines, grid.y_lines, (h, w))
    variants = []
    # Bounded nuisance model; sigma is measured in original source pixels.
    for sigma in (0.0, 0.5):
        reconstructed = rendered if sigma == 0 else ndimage.gaussian_filter(rendered, (sigma, sigma, 0), mode="nearest")
        distortion = _robust_error(evidence["premultiplied"], reconstructed)
        blur_penalty = 0.0004 if sigma else 0.0
        structural = structure_error(evidence, reconstructed)
        variants.append((distortion + blur_penalty + config.structure_weight * structural,
                         distortion, structural, sigma, blur_penalty))
    _, distortion, structural, sigma, blur_penalty = min(variants)
    complexity, parts = expression_complexity(cells.rgba, h * w)
    grid_cost = float(np.clip(grid.edge_score, 0, 1) + grid.warp_penalty)
    total = (distortion + blur_penalty + config.complexity_weight * complexity
             + config.structure_weight * structural + config.grid_weight * grid_cost)
    return {
        "total": float(total), "reconstruction": distortion,
        "complexity": complexity, "structure": structural, "grid": grid_cost,
        "blur_sigma": sigma, "blur_penalty": blur_penalty,
        "complexity_parts": parts,
        "weighted": {"complexity": config.complexity_weight * complexity,
                     "structure": config.structure_weight * structural,
                     "grid": config.grid_weight * grid_cost},
    }
