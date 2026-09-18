"""Optional color postprocessing. Never reads or changes the recovered grid."""
from dataclasses import dataclass, replace
from functools import lru_cache
import json
from pathlib import Path
from numbers import Integral
from time import perf_counter
import numpy as np
from PIL import Image
from .color_math import rgb_to_lab, distances, nearest

PALETTE_IDS = ("DMC436", "MARD24", "MARD48", "MARD72", "MARD96", "MARD120",
               "MARD144", "MARD221", "MARD280")
MAX_COLORS = 512


def validate_color_options(colors, palette, color_mode):
    if colors is not None and (isinstance(colors, bool) or not isinstance(colors, Integral)
                               or not 1 <= colors <= MAX_COLORS):
        raise ValueError(f"colors must be an integer from 1 to {MAX_COLORS}")
    if palette is not None and palette not in PALETTE_IDS:
        raise ValueError("unknown palette; choose " + ", ".join(PALETTE_IDS))
    if color_mode not in ("natural", "rgb"):
        raise ValueError("color_mode must be natural or rgb")


@lru_cache(maxsize=1)
def _libraries():
    return json.loads(Path(__file__).with_name("palettes.json").read_text(encoding="utf-8"))["palettes"]


def palette_catalog():
    return [dict(id=p["id"], name=f"拼豆-{p['brand']}-{p['nominal_size']}色",
                 brand=p["brand"], nominal_size=p["nominal_size"], entries=len(p["colors"]),
                 unique_colors=len({c["hex"].upper() for c in p["colors"]})) for p in _libraries()]


def palette_rgb(name):
    validate_color_options(None, name, "natural")
    if name is None:
        raise ValueError("a palette name is required")
    p = next(p for p in _libraries() if p["id"] == name)
    # Multiple bead IDs may have the same RGB; these are one image color.
    hexes = dict.fromkeys(c["hex"].upper().lstrip("#") for c in p["colors"])
    return np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in hexes], dtype=np.uint8)


def _space(rgb, mode):
    return rgb_to_lab(rgb) if mode == "natural" else np.asarray(rgb, dtype=np.float64)


def _representatives(rgb, weights, budget=2048):
    """Bound adaptive clustering using RGB bins, each represented by a real input color."""
    if len(rgb) <= budget:
        return rgb, weights
    for shift in (3, 4, 5):
        bins, inv = np.unique(rgb >> shift, axis=0, return_inverse=True)
        if len(bins) <= budget:
            break
    # Highest-support real color per bin, with original RGB order breaking ties.
    order = np.lexsort((np.arange(len(rgb)), -weights, inv))
    first = np.r_[True, inv[order][1:] != inv[order][:-1]]
    return rgb[order[first]], np.bincount(inv, weights=weights)


def _select_palette(rgb, weights, count, mode):
    """Deterministic weighted, constrained clustering; at most eight refinement rounds.

    Frequency/diversity seeds keep small contrasting colors eligible. Centers are
    snapped to real members; accept a move only if it lowers the selected metric.
    This is a bounded engineering heuristic, not an optimal K-color solution.
    """
    if len(rgb) <= count:
        return rgb
    points = _space(rgb, mode)
    selected = [int(np.argmax(weights))]
    best = distances(points, points[selected[0]], mode)
    for _ in range(1, count):
        priority = best * np.log1p(weights)
        priority[selected] = -1
        index = int(np.argmax(priority))
        selected.append(index)
        best = np.minimum(best, distances(points, points[index], mode))
    selected = np.array(selected)
    for _ in range(8):
        assignment = nearest(points, points[selected], mode)
        previous = selected.copy()
        for cluster in range(count):
            members = np.flatnonzero(assignment == cluster)
            if not len(members):
                continue
            mean = np.average(points[members], axis=0, weights=weights[members])
            candidate = members[np.argmin(distances(points[members], mean, mode))]
            old_cost = np.dot(weights[members], distances(points[members], points[selected[cluster]], mode))
            new_cost = np.dot(weights[members], distances(points[members], points[candidate], mode))
            if new_cost < old_cost - 1e-9:
                selected[cluster] = candidate
        if np.array_equal(previous, selected):
            break
    return rgb[np.unique(selected)]


@dataclass
class ColorResult:
    image: Image.Image
    diagnostics: dict
    seconds: float


def process_colors(image, *, colors=None, palette=None, color_mode="natural"):
    """Recolor an already restored image. Alpha and dimensions remain byte-exact.

    No library + no limit is an exact no-op. Palette matching and final remapping
    both use the requested distance. Fully transparent RGB has no statistical vote.
    """
    validate_color_options(colors, palette, color_mode)
    start = perf_counter()
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
    if colors is None and palette is None:
        return ColorResult(image.copy(), dict(applied=False, palette=None, limit=None,
                           mode=color_mode), perf_counter() - start)
    rgba = np.array(image.convert("RGBA"))
    visible = rgba[..., 3] > 0
    rgb, inv = np.unique(rgba[visible, :3], axis=0, return_inverse=True)
    count_before = len(rgb)
    # Fractional coverage has a fractional vote; no alpha threshold is introduced.
    weights = np.bincount(inv, weights=rgba[visible, 3].astype(float) / 255., minlength=len(rgb))
    if len(rgb):
        if palette:
            available = palette_rgb(palette)
            matched = nearest(_space(rgb, color_mode), _space(available, color_mode), color_mode)
            used, inverse = np.unique(matched, return_inverse=True)
            candidates = available[used]
            support = np.bincount(inverse, weights=weights, minlength=len(used))
            if colors is not None and len(candidates) > colors:
                reduced = _select_palette(candidates, support, colors, color_mode)
                remap = nearest(_space(candidates, color_mode), _space(reduced, color_mode), color_mode)
                mapped = reduced[remap[inverse]]
            else:
                mapped = available[matched]
        elif len(rgb) > colors:
            candidates, support = _representatives(rgb, weights)
            reduced = _select_palette(candidates, support, colors, color_mode)
            mapped = reduced[nearest(_space(rgb, color_mode), _space(reduced, color_mode), color_mode)]
        else:
            mapped = rgb
        rgba[visible, :3] = mapped[inv]
        count_after = len(np.unique(mapped, axis=0))
    else:
        count_after = 0
    rgba[~visible, :3] = 0
    result = Image.fromarray(rgba if image.mode == "RGBA" else rgba[..., :3])
    info = dict(applied=True, palette=palette, limit=colors, mode=color_mode,
                input_colors=count_before, output_colors=count_after)
    if palette:
        info["library"] = next(p for p in palette_catalog() if p["id"] == palette)
    return ColorResult(result, info, perf_counter() - start)


def quantize_cells(cells, colors, *, palette=None, color_mode="natural"):
    """Compatibility helper for callers processing float CellResult arrays."""
    from .image_io import to_pil
    result = process_colors(to_pil(cells.rgba), colors=colors, palette=palette, color_mode=color_mode)
    rgba = cells.rgba.copy()
    rgba[..., :3] = np.asarray(result.image)[..., :3] / 255.
    if palette is None and (colors is None or result.diagnostics["input_colors"] <= colors):
        rgba = cells.rgba.copy()
    rgba[rgba[..., 3] == 0, :3] = 0
    return replace(cells, rgba=rgba)
