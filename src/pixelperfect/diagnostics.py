"""Opt-in FFT, colour-edge and grid pictures, drawn with Pillow only."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image, ImageDraw


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


def _display(image, title, limit=1024, lines=None):
    source_width, source_height = image.size
    image = image.copy()
    image.thumbnail((limit, limit), Image.Resampling.NEAREST)
    frame = Image.new("RGB", (max(520, image.width), image.height + 42), "#20232b")
    left = (frame.width - image.width) // 2
    frame.paste(image, (left, 42))
    draw = ImageDraw.Draw(frame)
    # Draw after thumbnailing: a one-source-pixel line otherwise disappears
    # unpredictably under nearest-neighbour display reduction.
    if lines is not None:
        for x in lines[0]:
            px = left + min(image.width - 1, max(0, x * image.width / source_width))
            draw.line((px, 42, px, 41 + image.height), fill="#ff657c")
        for y in lines[1]:
            py = 42 + min(image.height - 1, max(0, y * image.height / source_height))
            draw.line((left, py, left + image.width - 1, py), fill="#ff657c")
    draw.text((12, 12), title, fill="white")
    return frame


def _plot(draw, rect, values, color, label, markers=()):
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline="#aaa")
    values = np.asarray(values)
    maximum = max(float(values.max(initial=0)), 1e-8)
    points = [(x0 + i * (x1 - x0) / max(len(values) - 1, 1),
               y1 - float(v) / maximum * (y1 - y0 - 22)) for i, v in enumerate(values)]
    if len(points) > 1:
        draw.line(points, fill=color, width=1)
    for m in markers:
        x = x0 + float(m) / max(len(values) - 1, 1) * (x1 - x0)
        if x0 <= x <= x1:
            draw.line((x, y0, x, y1), fill="#f16b7f", width=1)
    draw.text((x0 + 6, y0 + 4), label, fill="white")


def write_debug(result, original=None, directory=None, export_path=None, export_scale=1):
    started = perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    debug = result.debug_data
    draw_grid = not result.grid["fallback"] and not result.grid.get("native_preserved", False)
    w, h = result.grid["input_size"]
    half = debug["spectrum"]
    tail = half[:, 1:-1] if w % 2 == 0 else half[:, 1:]
    mirrored = tail[(-np.arange(h)) % h, ::-1]
    spectrum = np.fft.fftshift(np.concatenate((half, mirrored), axis=1))
    low, high = float(spectrum.min()), float(np.percentile(spectrum, 99.5))
    pixels = np.clip((spectrum - low) / max(high - low, 1e-8), 0, 1)
    fft = Image.fromarray(np.rint(pixels * 255).astype(np.uint8)).convert("RGB")
    fft_lines = None
    if draw_grid:
        fft_lines = ([w // 2 + offset * w / result.grid["sx"] for offset in (-1, 1)],
                     [h // 2 + offset * h / result.grid["sy"] for offset in (-1, 1)])
    _display(fft, "Image FFT: log(1+|F|); red = selected reciprocal spacing", lines=fft_lines).save(directory / "fft.png")
    ex, ey = debug["edge_x"], debug["edge_y"]
    # Independent x/y derivatives remain visible: x is red, y is green.
    strength = np.maximum(ex, ey)
    cap = max(float(np.percentile(strength[strength > 0], 95)), .05) if np.any(strength > 0) else 1.
    edge = np.stack((ex, ey, np.minimum(ex, ey)), axis=-1)
    edge = np.rint(np.clip(edge / cap, 0, 1) * 255).astype(np.uint8)
    _display(Image.fromarray(edge), "Colour + alpha edges: X=red, Y=green; source resolution").save(directory / "edges.png")
    source = debug["source"]
    # Neutral background is for display only; the result PNG keeps its alpha.
    bg = Image.new("RGBA", source.size, (190, 190, 190, 255))
    bg.alpha_composite(source)
    overlay = bg.convert("RGB")
    lines = (result.grid["x_lines"], result.grid["y_lines"]) if draw_grid else None
    title = (f"Grid {result.image.width}x{result.image.height}; "
             f"spacing {result.grid['sx']:.3f}x{result.grid['sy']:.3f}; confidence {result.confidence:.3f}")
    _display(overlay, title, lines=lines).save(directory / "grid.png")
    plots = Image.new("RGB", (1040, 620), "#20232b")
    d = ImageDraw.Draw(plots)
    for i, (profile, lines, name) in enumerate([
            (debug["profile_x"], result.grid["x_lines"], "X"),
            (debug["profile_y"], result.grid["y_lines"], "Y")]):
        y = i * 300
        _plot(d, (20, y + 10, 1015, y + 140), profile, "#8ddbc1", name + " colour edge projection; red=cuts",
              lines if draw_grid else ())
        power = abs(np.fft.rfft(profile - profile.mean()))
        spacing = result.grid["sx" if i == 0 else "sy"]
        _plot(d, (20, y + 160, 1015, y + 290), power, "#e2c87a", name + " edge-projection FFT",
              [len(profile) / spacing] if draw_grid else ())
    plots.save(directory / "profiles.png")
    knots = Image.new("RGB", (1040, 330), "#20232b")
    kd = ImageDraw.Draw(knots)
    for i, axis in enumerate(("x", "y")):
        cuts = np.asarray(result.grid[axis + "_lines"])
        centres = (cuts[:-1] + cuts[1:]) / 2 - .5
        _plot(kd, (20, 10+i*160, 1015, 150+i*160), debug["curvature_" + axis], "#8dbaf0",
              axis.upper()+" curvature; red=cell centres; model="+
              result.diagnostics["grid_search"].get("evidence_model", "boundaries"),
              centres if draw_grid else ())
    knots.save(directory / "curvature.png")
    result.timings["debug_images"] = perf_counter() - started
    result.timings["total_with_export"] = (result.timings["total"] + result.timings.get("save_png", 0)
                                           + result.timings["debug_images"])
    report = dict(grid=result.grid, heuristic_confidence=result.confidence,
                  timings_seconds=result.timings, diagnostics=result.diagnostics,
                  export=dict(path=str(export_path) if export_path else None, scale=export_scale,
                              size=[result.image.width * export_scale, result.image.height * export_scale]))
    (directory / "info.json").write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False,
                                                   allow_nan=False), encoding="utf-8")
    summary = [
        f"Native output grid: {result.image.width} x {result.image.height}",
        f"Source pixel spacing: {result.grid['sx']:.6f} x {result.grid['sy']:.6f}",
        f"Initial phase: {result.grid['phase_x']:.6f}, {result.grid['phase_y']:.6f}",
        f"Median actual cell size: {result.grid['median_cell_width']} x {result.grid['median_cell_height']}",
        f"Selected by: {result.grid['source']}; local adjustment: {result.grid['warped']}",
        f"Heuristic confidence (NOT probability): {result.confidence:.4f}",
        f"Fallback: {result.grid['fallback']}; export scale: {export_scale}",
        f"Alpha policy: {result.diagnostics.get('structure', {}).get('alpha_mode', 'unchanged fallback')}",
        f"Centre impulses rejected: {result.diagnostics.get('structure', {}).get('rejected_central_impulses', 0)}; "
        "contour expansion: disabled",
        f"Processing: {result.timings['total']:.4f}s; with PNG/debug: {result.timings['total_with_export']:.4f}s",
        "All source pixels are covered. Display thumbnails do not affect detection.",
        *result.diagnostics["warnings"],
    ]
    (directory / "info.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
