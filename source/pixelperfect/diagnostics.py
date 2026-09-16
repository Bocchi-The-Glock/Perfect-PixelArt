"""Explicit opt-in diagnostics; no matplotlib or extra runtime dependency."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .image_io import load_image, to_pil
from .scoring import render_grid


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_debug(result, original, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = {"grid": result.grid, "heuristic_confidence": result.confidence,
              "timings_seconds": result.timings, "diagnostics": result.diagnostics}
    (directory / "candidates.json").write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2,
                                                        allow_nan=False), encoding="utf-8")
    source = load_image(original)
    overlay = to_pil(source.rgba, True)
    draw = ImageDraw.Draw(overlay)
    for x in result.grid["x_lines"]:
        draw.line((float(x), 0, float(x), overlay.height), fill=(255, 60, 80, 210), width=1)
    for y in result.grid["y_lines"]:
        draw.line((0, float(y), overlay.width, float(y)), fill=(255, 60, 80, 210), width=1)
    overlay.save(directory / "grid_overlay.png")
    c = np.clip(result.cell_confidence, 0, 1)
    color = np.stack((1 - c, c, np.zeros_like(c)), axis=-1)
    Image.fromarray(np.rint(color * 255).astype(np.uint8)).save(directory / "cell_confidence.png")
    low = load_image(result.image).rgba
    p = render_grid(low, result.grid["x_lines"], result.grid["y_lines"], source.rgba.shape[:2])
    rgba = p.copy()
    np.divide(p[..., :3], p[..., 3:4], out=rgba[..., :3], where=p[..., 3:4] > 1e-8)
    rgba[p[..., 3] <= 1e-8, :3] = 0
    to_pil(rgba, source.has_alpha).save(directory / "reconstruction.png")
