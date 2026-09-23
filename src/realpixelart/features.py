"""Original-resolution colour edges and Fourier evidence, using NumPy only."""
from dataclasses import dataclass
import numpy as np

DENSE_PIXEL_LIMIT = 4_000_000


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


def axis_segment_evidence(rgba, spacing):
    """Measure straight, axis-aligned edge runs on at most nine source patches.

    Original pixels are never resized. Centred, tangentially smoothed derivatives
    distinguish a diagonal staircase from an actual horizontal/vertical segment.
    This is supporting evidence, not a pixel-art classifier.
    """
    h, w = rgba.shape[:2]
    ph, pw = min(h, 130), min(w, 130)
    if min(ph, pw) < 5:
        return dict(active_patches=0, sampled_pixels=0, patches=[])
    yy = np.unique(np.linspace(0, h - ph, min(3, max(1, h // ph))).astype(int))
    xx = np.unique(np.linspace(0, w - pw, min(3, max(1, w // pw))).astype(int))
    origins = [(int(x), int(y)) for y in yy for x in xx]
    values = np.stack([rgba[y:y+ph, x:x+pw] for x, y in origins])
    values[..., :3] *= values[..., 3:]
    gx = np.zeros((len(origins), ph-2, pw-2), np.float32)
    gy = np.zeros_like(gx)
    for channel in range(4):
        v = values[..., channel]
        dx = (v[:, :, 2:] - v[:, :, :-2]) * .5
        dy = (v[:, 2:, :] - v[:, :-2, :]) * .5
        np.maximum(gx, abs((dx[:, :-2] + 2*dx[:, 1:-1] + dx[:, 2:]) * .25), out=gx)
        np.maximum(gy, abs((dy[:, :, :-2] + 2*dy[:, :, 1:-1] + dy[:, :, 2:]) * .25), out=gy)
    strength = np.maximum(gx, gy)
    threshold = np.maximum(.012, np.minimum(.04, .15*strength.max(axis=(1, 2))))[:, None, None]
    strong = strength >= threshold
    vertical = strong & (gx >= 3*gy)
    horizontal = strong & (gy >= 3*gx)

    def sustained(mask, length, axis):
        padding = [(0, 0)] * 3
        padding[axis] = (length // 2 + 1, (length - 1) // 2)
        total = np.cumsum(np.pad(mask, padding), axis=axis, dtype=np.int32)
        first, last = [slice(None)]*3, [slice(None)]*3
        first[axis], last[axis] = slice(None, -length), slice(length, None)
        count = total[tuple(last)] - total[tuple(first)]
        return mask & (count >= length - (1 if length >= 6 else 0))

    lengths = [max(3, min(16, round(.5*s))) for s in spacing]
    vr = sustained(vertical, lengths[1], 1)
    hr = sustained(horizontal, lengths[0], 2)
    mass = np.minimum(strength, .2) * strong
    patches = []
    for index, origin in enumerate(origins):
        count = int(strong[index].sum())
        weight = float(mass[index].sum())
        denom = max(weight, 1e-9)
        patches.append(dict(origin=list(origin), edges=count, mass=weight,
                            vertical=float(mass[index][vr[index]].sum() / denom),
                            horizontal=float(mass[index][hr[index]].sum() / denom),
                            axis_aligned=float(mass[index][vertical[index] | horizontal[index]].sum() / denom)))
    active = [p for p in patches if p['edges'] >= 32 and p['mass'] >= 1.]
    return dict(sampled_pixels=int(values.shape[0]*ph*pw), patch_size=[pw, ph],
                run_length=lengths, active_patches=len(active), patches=patches,
                segment_fraction=float(np.mean([p['vertical']+p['horizontal'] for p in active])) if active else 0.,
                axis_fraction=float(np.mean([p['axis_aligned'] for p in active])) if active else 0.)


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
    native_axes: tuple
    mode: str = 'full image'
    spectrum_size: tuple | None = None
    preview_edges: tuple | None = None


def _native_axis(scanlines):
    """Count one-pixel contrast reversals on original-resolution alpha-aware strips.

    A-B-C votes only where B leaves the interval between A and C in at least one
    premultiplied channel. Smooth monotone edge ramps do not vote.
    """
    values = scanlines.copy()
    values[..., :3] *= values[..., 3:]
    delta = np.diff(values, axis=1)
    edges = np.max(np.abs(delta), axis=-1) > .06
    reversal = np.max((np.abs(delta[:, :-1]) + np.abs(delta[:, 1:])
                       - np.abs(delta[:, :-1] + delta[:, 1:])) * .5, axis=-1) > .06
    counts = reversal.sum(axis=1)
    return dict(edge_count=int(edges.sum()), turn_count=int(counts.sum()),
                turn_fraction=float(counts.sum() / max(1, edges.sum())),
                supporting_lines=int(np.count_nonzero(counts >= 2)),
                active_lines=int(np.count_nonzero(edges.sum(axis=1) >= 4)))


def _log_spectrum(gray, block=64):
    """Exact separable 2-D FFT, with bounded transform temporaries.

    Keep complex128 precision and every source pixel. NumPy's rfft2 otherwise
    holds multiple image-sized complex arrays during its second transform.
    """
    h, w = gray.shape
    horizontal = np.empty((h, w // 2 + 1), np.complex128)
    for start in range(0, h, block):
        horizontal[start:start + block] = np.fft.rfft(gray[start:start + block], axis=1)
    # Match rfft2's column-major result, including the reduction order used by
    # spectral_x/y. Changing layout can otherwise perturb float32 mean scores.
    spectrum = np.empty(horizontal.shape, np.float32, order='F')
    for start in range(0, horizontal.shape[1], block):
        transformed = np.fft.fft(horizontal[:, start:start + block], axis=0)
        magnitude = np.abs(transformed)
        del transformed
        np.log1p(magnitude, out=magnitude)
        spectrum[:, start:start + block] = magnitude
    return spectrum


def _scanline_features(scanlines):
    """Full-length source lines: their spacing remains in original pixels."""
    values = scanlines.copy()
    values[..., :3] *= values[..., 3:]
    gradient = np.zeros(values.shape[:2], np.float32)
    curvature = np.zeros_like(gradient)
    gray = np.zeros_like(gradient)
    for c, weight in enumerate((.299, .587, .114, 0.)):
        v = values[..., c]
        np.maximum(gradient[:, 1:], abs(np.diff(v, axis=1)), out=gradient[:, 1:])
        np.maximum(curvature[:, 1:-1], abs(np.diff(v, n=2, axis=1)), out=curvature[:, 1:-1])
        gray += weight * v
    gray += .5 * (1 - values[..., 3])
    spectral = np.log1p(abs(np.fft.rfft(gray, axis=1))).mean(axis=0).astype(np.float32)
    return gradient, smooth(np.minimum(gradient, .35).mean(axis=0)), spectral, \
        smooth(curvature.mean(axis=0)), float(curvature.sum() / max(float(gradient.sum()), 1e-9))


def _sparse_features(rgba):
    h, w = rgba.shape[:2]
    rows = np.unique(np.linspace(0, h - 1, min(96, h)).astype(int))
    cols = np.unique(np.linspace(0, w - 1, min(96, w)).astype(int))
    horizontal, vertical = rgba[rows], rgba[:, cols].transpose(1, 0, 2)
    gx, px, sx, cx, rx = _scanline_features(horizontal)
    gy, py, sy, cy, ry = _scanline_features(vertical)
    axes = (_native_axis(horizontal), _native_axis(vertical))
    del horizontal, vertical
    # A genuine 2-D FFT of a bounded source patch is for visualization only.
    # Candidate periods come from the full-length source lines above.
    ph, pw = min(h, 512), min(w, 512)
    patch = rgba[(h-ph)//2:(h+ph)//2, (w-pw)//2:(w+pw)//2]
    gray = np.zeros((ph, pw), np.float32)
    for c, weight in enumerate((.299, .587, .114)):
        gray += weight * (patch[..., c] * patch[..., 3])
    gray += .5 * (1 - patch[..., 3])
    spectrum = _log_spectrum(gray)
    # Display original derivatives at bounded preview positions, not derivatives
    # of a resized image. This does not feed the detector.
    ratio = min(1., 1024 / max(h, w))
    nx, ny = max(1, round(w * ratio)), max(1, round(h * ratio))
    xx = ((np.arange(nx) + .5) * w / nx).astype(int)
    yy = ((np.arange(ny) + .5) * h / ny).astype(int)
    centre = rgba[np.ix_(yy, xx)].copy()
    left = rgba[np.ix_(yy, np.maximum(xx-1, 0))].copy()
    top = rgba[np.ix_(np.maximum(yy-1, 0), xx)].copy()
    for samples in (centre, left, top):
        samples[..., :3] *= samples[..., 3:]
    ex, ey = np.max(abs(centre-left), axis=2), np.max(abs(centre-top), axis=2)
    return FeatureData(gx, gy.T, px, py, spectrum, sx, sy, cx, cy, (rx, ry), axes,
                       mode='96 original-resolution scanlines per axis', spectrum_size=(pw, ph), preview_edges=(ex, ey))


def extract_features(rgba):
    if rgba.shape[0] * rgba.shape[1] > DENSE_PIXEL_LIMIT:
        return _sparse_features(rgba)
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
    # Compute the same full-resolution gradients in stripes. The one-row halo
    # retains derivatives at stripe boundaries; no resize or sampled grid signal.
    for start in range(0, len(alpha), 64):
        end = min(start + 64, len(alpha))
        top = max(0, start - 1)
        a = alpha[top:end]
        for channel, weight in enumerate((.299, .587, .114, 0.)):
            v = rgba[top:end, :, channel] * a if channel < 3 else a
            local = v[start - top:]
            target = gx[start:end, 1:]
            np.maximum(target, np.abs(np.diff(local, axis=1)), out=target)
            target = gy[max(1, start):end]
            np.maximum(target, np.abs(np.diff(v, axis=0)), out=target)
            gray[start:end] += weight * local
        gray[start:end] += .5 * (1 - alpha[start:end])
    # Premultiplied original-resolution strips retain the curvature evidence.
    for channel, weight in enumerate((.299, .587, .114, 0.)):
        vx = rgba[rows, :, channel] * alpha[rows] if channel < 3 else alpha[rows]
        vy = rgba[:, cols, channel] * alpha[:, cols] if channel < 3 else alpha[:, cols]
        cx[:, 1:-1] = np.maximum(cx[:, 1:-1], np.abs(np.diff(vx, n=2, axis=1)))
        cy[1:-1] = np.maximum(cy[1:-1], np.abs(np.diff(vy, n=2, axis=0)))
    # Reuse this real-input 2-D FFT for detection and the diagnostic image.
    spectrum = _log_spectrum(gray)
    del gray
    sx = spectrum[1:].mean(axis=0) if len(spectrum) > 1 else spectrum[0]
    sy = spectrum[:, 1:].mean(axis=1) if spectrum.shape[1] > 1 else spectrum[:, 0]
    px = smooth(np.minimum(gx, .35).mean(axis=0))
    py = smooth(np.minimum(gy, .35).mean(axis=1))
    ratio = (float(cx.sum() / max(float(gx[rows].sum()), 1e-9)),
             float(cy.sum() / max(float(gy[:, cols].sum()), 1e-9)))
    return FeatureData(gx, gy, px, py, spectrum, sx, sy[:len(sy) // 2 + 1],
                       smooth(cx.mean(axis=0)), smooth(cy.mean(axis=1)), ratio,
                       (_native_axis(rgba[rows]), _native_axis(rgba[:, cols].transpose(1, 0, 2))))
