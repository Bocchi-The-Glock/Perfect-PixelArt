"""Browser adapter. All image decisions remain inside pixelperfect, unchanged."""
import json
import platform
from pathlib import Path
import zipfile
import numpy as np
from PIL import __version__ as pillow_version
from PIL import Image
from pixelperfect import Config, pixelize, save_result, export_png, process_colors


def process(input_path, request_json, output_directory):
    request = json.loads(request_json)
    settings = request.get("config", {})
    config = Config(**settings)
    debug = bool(request.get("debug", False))
    stem = Path(request.get("name", "image.png")).stem
    stem = "".join(c for c in stem if c not in '/\\:*?"<>|' and ord(c) >= 32).strip(". ") or "image"
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result = pixelize(input_path, config)
    save_result(result, output / "result.png", scale=config.scale,
                debug=debug, debug_dir=output / "debug" if debug else None)
    result.image.save(output / "native.png")
    result.native_image.save(output / "base.png")
    # Reuse the EXIF-corrected input; never feed canvas-decoded pixels to Python.
    result.debug_data["source"].save(output / "original.png")
    if debug:
        with zipfile.ZipFile(output / "debug.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((output / "debug").iterdir()):
                archive.write(path, f"output/debug/{stem}/{path.name}")
    return json.dumps(dict(name=stem + ".png", grid=result.grid, confidence=result.confidence,
                           timings=result.timings, warnings=result.diagnostics["warnings"],
                           config=settings, debug=debug, color_processing=result.diagnostics["color_processing"],
                           export_size=[result.image.width * config.scale, result.image.height * config.scale],
                           runtime=dict(python=platform.python_version(), numpy=np.__version__, pillow=pillow_version)))


def export_native(input_path, output_path, scale):
    """Resize cached native PNG with the exact same desktop export function."""
    with Image.open(input_path) as image:
        export_png(image, output_path, scale)


def recolor_native(input_path, output_path, settings_json, debug_path=None):
    """Independent optional second stage; always consumes the unlimited native PNG."""
    with Image.open(input_path) as image:
        result = process_colors(image, **json.loads(settings_json))
    result.image.save(output_path)
    if debug_path is not None:
        with zipfile.ZipFile(debug_path) as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
        for name in entries:
            if name.endswith('/info.json'):
                info = json.loads(entries[name])
                info['diagnostics']['color_processing'] = result.diagnostics
                timings = info['timings_seconds']
                difference = result.seconds - timings.get('palette', 0)
                for key in ('total', 'total_with_export'):
                    timings[key] += difference
                timings['palette'] = result.seconds
                entries[name] = json.dumps(info, ensure_ascii=False, indent=2).encode('utf-8')
            elif name.endswith('/info.txt'):
                lines = entries[name].decode('utf-8').splitlines()
                lines = [line for line in lines if not line.startswith(('Processing:', 'Color postprocessing:'))]
                lines.append('Color postprocessing: ' + json.dumps(result.diagnostics) + f'; {result.seconds:.4f}s')
                entries[name] = ('\n'.join(lines) + '\n').encode('utf-8')
        with zipfile.ZipFile(debug_path, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, content in entries.items():
                archive.writestr(name, content)
    return json.dumps(dict(color_processing=result.diagnostics, seconds=result.seconds))
