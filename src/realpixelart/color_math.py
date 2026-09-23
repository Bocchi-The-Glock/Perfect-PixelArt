"""sRGB/D65 Lab and CIEDE2000 (kL=kC=kH=1), using only NumPy.

Equations/reference pairs: Sharma, Wu & Dalal (2005),
https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/
Inputs to rgb_to_lab are encoded RGB in [0,255]. No display/ICC calibration.
"""
import numpy as np


def rgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=np.float64) / 255.
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    xyz = linear @ np.array([[.4124564, .2126729, .0193339],
                             [.3575761, .7151522, .1191920],
                             [.1804375, .0721750, .9503041]])
    xyz /= [.95047, 1., 1.08883]
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    x, y, z = np.moveaxis(f, -1, 0)
    return np.stack((116 * y - 16, 500 * (x - y), 200 * (y - z)), axis=-1)


def delta_e_2000(first, second):
    """Broadcast arrays ending in three Lab components; return perceptual distance."""
    l1, a1, b1 = np.moveaxis(np.asarray(first, dtype=np.float64), -1, 0)
    l2, a2, b2 = np.moveaxis(np.asarray(second, dtype=np.float64), -1, 0)
    cbar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = .5 * (1 - np.sqrt(cbar ** 7 / (cbar ** 7 + 25 ** 7)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    c1, c2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    h1 = np.mod(np.degrees(np.arctan2(b1, ap1)), 360)
    h2 = np.mod(np.degrees(np.arctan2(b2, ap2)), 360)
    zero = c1 * c2 == 0
    dh = h2 - h1
    # Tolerance makes exactly opposite hues stable under floating-point rounding.
    dh = np.where(dh > 180 + 1e-12, dh - 360, np.where(dh < -180 - 1e-12, dh + 360, dh))
    dh = np.where(zero, 0, dh)
    dl, dc = l2 - l1, c2 - c1
    dh_term = 2 * np.sqrt(c1 * c2) * np.sin(np.radians(dh / 2))
    lm, cm = (l1 + l2) / 2, (c1 + c2) / 2
    hm = np.where(np.abs(h1 - h2) <= 180 + 1e-12, (h1 + h2) / 2,
                  np.where(h1 + h2 < 360, (h1 + h2 + 360) / 2, (h1 + h2 - 360) / 2))
    hm = np.where(zero, h1 + h2, hm)
    t = (1 - .17 * np.cos(np.radians(hm - 30)) + .24 * np.cos(np.radians(2 * hm))
         + .32 * np.cos(np.radians(3 * hm + 6)) - .20 * np.cos(np.radians(4 * hm - 63)))
    sl = 1 + .015 * (lm - 50) ** 2 / np.sqrt(20 + (lm - 50) ** 2)
    sc, sh = 1 + .045 * cm, 1 + .015 * cm * t
    rt = -2 * np.sqrt(cm ** 7 / (cm ** 7 + 25 ** 7)) * np.sin(
        np.radians(60 * np.exp(-((hm - 275) / 25) ** 2)))
    vl, vc, vh = dl / sl, dc / sc, dh_term / sh
    return np.sqrt(np.maximum(0, vl * vl + vc * vc + vh * vh + rt * vc * vh))


def distances(first, second, mode):
    """Squared distances; inputs already transformed to the selected space."""
    if mode == "natural":
        return delta_e_2000(first, second) ** 2
    return np.sum((first - second) ** 2, axis=-1)


def nearest(points, palette, mode):
    """Exact full-palette search, bounded temporary memory; stable first-index ties."""
    result = np.empty(len(points), dtype=np.int32)
    chunk = max(32, 32768 // max(1, len(palette)))
    for start in range(0, len(points), chunk):
        result[start:start + chunk] = np.argmin(
            distances(points[start:start + chunk, None], palette[None], mode), axis=1)
    return result
