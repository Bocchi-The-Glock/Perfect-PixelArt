"""Bounded deterministic palette fitting on recovered cells, without dithering."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .sampling import CellResult, _detail_mask


def _nearest(values, centers):
    labels = np.empty(len(values), np.int32)
    for start in range(0, len(values), 4096):
        block = values[start:start + 4096]
        labels[start:start + len(block)] = ((block[:, None] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
    return labels


def quantize_cells(cells: CellResult, colors: int) -> CellResult:
    """Fit <= colors visible RGB values; alpha is preserved independently.

    Training is capped at 8192 low-resolution cells and 10 Lloyd iterations.
    Weights account for alpha, recovery confidence, local contrast and rarity.
    Up to a quarter of palette entries retain salient detail colours exactly;
    such a small budget cannot guarantee every detail keeps a distinct colour.
    """
    if isinstance(colors, bool) or not isinstance(colors, (int, np.integer)) or not 1 <= colors <= 256:
        raise ValueError("colors must be an integer from 1 to 256")
    output = cells.rgba.copy()
    flat = output.reshape(-1, 4)
    valid = flat[:, 3] > 1e-8
    flat[~valid, :3] = 0
    rgb = flat[valid, :3]
    info = dict(cells.structure)
    if len(rgb) == 0:
        info.update(palette_size=0, palette_requested=int(colors), palette_detail_changes=0)
        return replace(cells, rgba=output, structure=info)
    unique = np.unique(rgb, axis=0)
    if len(unique) <= colors:
        info.update(palette_size=int(len(unique)), palette_requested=int(colors), palette_detail_changes=0)
        return replace(cells, rgba=output, structure=info)
    salience = _detail_mask(output).ravel()[valid]
    confidence = cells.confidence.ravel()[valid]
    weights = flat[valid, 3] * (.25 + .75 * confidence) * (1 + np.minimum(salience, 2))
    details = cells.detail_mask.ravel()[valid] if cells.detail_mask is not None else salience > 1.0
    # RGB bins estimate rarity spatially weighted by boundary contrast, rather
    # than using colour histogram entropy as a reconstruction score.
    bins = np.minimum((rgb * 15).astype(np.int32), 14)
    keys = bins[:, 0] * 225 + bins[:, 1] * 15 + bins[:, 2]
    counts = np.bincount(keys, minlength=3375)
    rarity = 1 / np.sqrt(counts[keys])
    priorities = salience * rarity * np.sqrt(confidence + .1)
    reserve_limit = min(colors // 4, 8)
    reserved = []
    if reserve_limit:
        for index in np.argsort(-priorities, kind="stable"):
            if details[index] and all(np.linalg.norm(rgb[index] - c) > .15 for c in reserved):
                reserved.append(rgb[index].copy())
            if len(reserved) >= reserve_limit:
                break
    train_indices = np.unique(np.concatenate((
        np.linspace(0, len(rgb) - 1, min(7168, len(rgb))).astype(int),
        np.argsort(-priorities, kind="stable")[:min(1024, len(rgb))])))
    training = rgb[train_indices]
    train_weights = weights[train_indices] * (1 + .5 * rarity[train_indices])
    centers = list(reserved)
    if not centers:
        centers.append(training[int(np.argmax(train_weights))].copy())
    distances = np.full(len(training), np.inf)
    for center in centers:
        distances = np.minimum(distances, ((training - center) ** 2).sum(axis=1))
    while len(centers) < colors:
        index = int(np.argmax(distances * np.sqrt(train_weights)))
        if distances[index] <= 1e-12:
            break
        center = training[index].copy()
        centers.append(center)
        distances = np.minimum(distances, ((training - center) ** 2).sum(axis=1))
    centers = np.asarray(centers, np.float32)
    for _ in range(10):
        labels = _nearest(training, centers)
        updated = centers.copy()
        for index in range(len(reserved), len(centers)):
            mask = labels == index
            if np.any(mask):
                updated[index] = np.average(training[mask], axis=0, weights=train_weights[mask])
        if np.max(np.abs(updated - centers)) < 1e-5:
            centers = updated
            break
        centers = updated
    labels = _nearest(rgb, centers)
    quantized = np.clip(centers[labels], 0, 1)
    detail_changes = int(np.sum(details & (np.linalg.norm(quantized - rgb, axis=1) > .15)))
    flat[valid, :3] = quantized
    info.update(palette_size=int(len(np.unique(quantized, axis=0))), palette_requested=int(colors),
                palette_detail_changes=detail_changes, palette_reserved_details=len(reserved))
    return replace(cells, rgba=output, structure=info)
