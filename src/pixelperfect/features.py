"""Original-resolution colour edges and Fourier evidence, using NumPy only."""
from dataclasses import dataclass
import numpy as np


def smooth(values, radius=1):
    kernel = np.array([1., 2., 1.]) if radius == 1 else np.ones(2 * radius + 1)
    return np.convolve(np.pad(values, (radius, radius), mode="edge"), kernel / kernel.sum(), "valid")


def peaks(values, threshold=0.):
    """Represent flat local maxima by their midpoint."""
    if len(values) < 3:
        return np.empty(0, int)
    changes = np.flatnonzero(np.diff(values) != 0) + 1
    starts, ends = np.r_[0, changes], np.r_[changes, len(values)]
    inside = (starts > 0) & (ends < len(values))
    starts, ends = starts[inside], ends[inside]
    valid = ((values[starts] > values[starts - 1]) & (values[starts] > values[ends])
             & (values[starts] >= threshold))
    return ((starts[valid] + ends[valid] - 1) // 2).astype(int)


@dataclass
class FeatureData:
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    profile_x: np.ndarray
    profile_y: np.ndarray
    spectrum: np.ndarray
    spectral_x: np.ndarray
    spectral_y: np.ndarray
    curvature_x: np.ndarray
    curvature_y: np.ndarray
    ramp_ratio: tuple


def extract_features(rgba):
    alpha = rgba[..., 3]
    gx = np.zeros(alpha.shape, np.float32)
    gy = np.zeros_like(gx)
    gray = np.zeros_like(gx)
    # Original-resolution scanlines: second derivatives locate interpolation
    # knots when wide bilinear ramps have no sharp first-derivative maximum.
    rows = np.unique(np.linspace(0, len(alpha)-1, min(64, len(alpha))).astype(int))
    cols = np.unique(np.linspace(0, alpha.shape[1]-1, min(64, alpha.shape[1])).astype(int))
    cx = np.zeros((len(rows), alpha.shape[1]), np.float32)
    cy = np.zeros((alpha.shape[0], len(cols)), np.float32)
    # Premultiply RGB, and include alpha itself to detect black silhouettes.
    for channel, weight in enumerate((.299, .587, .114, 0.)):
        v = rgba[..., channel] * alpha if channel < 3 else alpha
        gx[:, 1:] = np.maximum(gx[:, 1:], np.abs(np.diff(v, axis=1)))
        gy[1:] = np.maximum(gy[1:], np.abs(np.diff(v, axis=0)))
        cx[:, 1:-1] = np.maximum(cx[:, 1:-1], np.abs(np.diff(v[rows], n=2, axis=1)))
        cy[1:-1] = np.maximum(cy[1:-1], np.abs(np.diff(v[:, cols], n=2, axis=0)))
        gray += weight * v
    gray += .5 * (1 - alpha)
    # Reuse this real-input 2-D FFT for detection and the diagnostic image.
    spectrum = np.log1p(np.abs(np.fft.rfft2(gray))).astype(np.float32)
    sx = spectrum[1:].mean(axis=0) if len(spectrum) > 1 else spectrum[0]
    sy = spectrum[:, 1:].mean(axis=1) if spectrum.shape[1] > 1 else spectrum[:, 0]
    px = smooth(np.minimum(gx, .35).mean(axis=0))
    py = smooth(np.minimum(gy, .35).mean(axis=1))
    ratio = (float(cx.sum() / max(float(gx[rows].sum()), 1e-9)),
             float(cy.sum() / max(float(gy[:, cols].sum()), 1e-9)))
    return FeatureData(gx, gy, px, py, spectrum, sx, sy[:len(sy) // 2 + 1],
                       smooth(cx.mean(axis=0)), smooth(cy.mean(axis=1)), ratio)
