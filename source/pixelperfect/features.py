"""Alpha-aware directional evidence at the original image resolution.

Distances are Euclidean distances in normalized, premultiplied sRGB plus alpha.
They are inexpensive edge cues, not a perceptually uniform colour difference.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks


@dataclass
class FeatureData:
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    raw_gradient_x: np.ndarray
    raw_gradient_y: np.ndarray
    profiles_x: np.ndarray
    profiles_y: np.ndarray
    profile_x: np.ndarray
    profile_y: np.ndarray
    active_x: np.ndarray
    active_y: np.ndarray
    edge_positions_x: np.ndarray
    edge_positions_y: np.ndarray
    edge_weights_x: np.ndarray
    edge_weights_y: np.ndarray


def _differences(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.zeros(values.shape[:2], np.float32)
    gy = np.zeros_like(gx)
    gx[:, 1:] = np.sqrt(np.mean(np.diff(values, axis=1) ** 2, axis=2))
    gy[1:, :] = np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=2))
    return gx, gy


def _profiles(gradient: np.ndarray, direction: int):
    """Equal-vote strips; zero/transparent strips receive no normalization vote."""
    cross_size = gradient.shape[0 if direction == 0 else 1]
    n = min(6, max(1, cross_size // 16))
    strips = np.array_split(gradient, n, axis=0 if direction == 0 else 1)
    result, active = [], []
    for strip in strips:
        # Clip individual strong edges before projecting, then limit each strip
        # to one normalized vote. No resize is involved.
        positive = strip[strip > 1e-5]
        cap = max(float(np.percentile(positive, 85)), 0.015) if positive.size else 0.015
        projection = np.mean(np.minimum(strip, cap), axis=0 if direction == 0 else 1)
        peak = float(projection.max(initial=0))
        usable = peak >= 0.006 and np.count_nonzero(projection > peak * .15) >= 2
        active.append(usable)
        if usable:
            nonzero = projection[projection > peak * .1]
            norm = max(float(np.percentile(nonzero, 80)), 1e-5)
            result.append(np.minimum(projection / norm, 1.5).astype(np.float32))
        else:
            result.append(np.zeros_like(projection, dtype=np.float32))
    profiles = np.stack(result)
    active = np.asarray(active, dtype=bool)
    aggregate = profiles[active].mean(axis=0) if active.any() else profiles[0].copy()
    if aggregate.size > 2:
        peaks, props = find_peaks(aggregate, prominence=max(.035, float(aggregate.max()) * .09))
        weights = props["prominences"].astype(np.float64)
        # Bound phase-search cost while retaining positions across the image.
        if len(peaks) > 768:
            keep = np.sort(np.argsort(weights, kind="stable")[-768:])
            peaks, weights = peaks[keep], weights[keep]
    else:
        peaks, weights = np.array([], dtype=int), np.array([], dtype=float)
    return profiles, active, aggregate, peaks.astype(float), weights


def extract_features(rgba: np.ndarray) -> FeatureData:
    if rgba.ndim != 3 or rgba.shape[2] != 4 or min(rgba.shape[:2]) < 1:
        raise ValueError("features require a nonempty H x W x 4 RGBA array")
    alpha = rgba[..., 3:4]
    values = np.concatenate((rgba[..., :3] * alpha, alpha), axis=2).astype(np.float32)
    raw_x, raw_y = _differences(values)
    # Light detection-only smoothing in premultiplied coordinates avoids dark
    # RGB from transparent pixels. Original gradients remain available.
    smooth = gaussian_filter(values, sigma=(.45, .45, 0), mode="nearest")
    gx, gy = _differences(smooth)
    px, ax, ex, ix, wx = _profiles(gx, 0)
    py, ay, ey, iy, wy = _profiles(gy, 1)
    return FeatureData(gx, gy, raw_x, raw_y, px, py, ex, ey, ax, ay, ix, iy, wx, wy)
