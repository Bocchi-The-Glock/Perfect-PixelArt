"""Optional Pillow palette; hidden transparent RGB never uses a colour slot."""
from dataclasses import replace
import numpy as np
from PIL import Image


def quantize_cells(cells, colors):
    if isinstance(colors, bool) or not isinstance(colors, (int, np.integer)) or not 1 <= colors <= 256:
        raise ValueError("colors must be an integer from 1 to 256")
    output = cells.rgba.copy()
    output[output[...,3] == 0,:3] = 0
    visible = output[..., 3] > 0
    if not visible.any():
        return replace(cells, rgba=output)
    if len(np.unique(output[visible,:3],axis=0)) <= colors:
        return replace(cells, rgba=output)
    rgb = np.rint(np.clip(output[visible, :3], 0, 1) * 255).astype(np.uint8)
    packed = Image.fromarray(rgb[None])
    quantized = packed.quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                               dither=Image.Dither.NONE).convert("RGB")
    output[visible, :3] = np.asarray(quantized)[0] / 255.
    return replace(cells, rgba=output)
