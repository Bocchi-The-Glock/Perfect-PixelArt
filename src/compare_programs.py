"""Run the two original programs on real files; each detects its own grid."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def load_main():
    directory = ROOT.parent / "perfectPixel-main/src/perfect_pixel"
    name = "_comparison_main"
    spec = importlib.util.spec_from_file_location(
        name, directory / "__init__.py", submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.get_perfect_pixel


def white_input(path):
    """Prepare one common RGB input, preserving visible alpha coverage on white."""
    with Image.open(path) as image:
        rgba = ImageOps.exif_transpose(image).convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")


def run_pair(path, destination, main_function, debug_dir=None):
    name = path.stem + ".png"
    main_path, real_path = destination/"main"/name, destination/"realpixelart"/name
    shared_path = destination/"input_white"/name
    if path.resolve() in (main_path.resolve(), real_path.resolve(), shared_path.resolve()):
        raise ValueError("Comparison output must not overwrite its input")
    main_path.parent.mkdir(parents=True, exist_ok=True)
    real_path.parent.mkdir(parents=True, exist_ok=True)
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    prepared = white_input(path)
    prepared.save(shared_path)
    rgb = np.array(prepared)
    record = {"input": str(path),
              "shared_input": {"path": str(shared_path), "background": [255, 255, 255],
                               "mode": "RGB", "size": list(prepared.size),
                               "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()},
              "main": {"backend": main_function.__module__},
              "realpixelart": {"entry": str(ROOT/"realpixelart.py")}}
    main_image = real_image = None
    # Remove only this runner's previous result files so failures cannot reuse them.
    main_path.unlink(missing_ok=True)
    real_path.unlink(missing_ok=True)
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            width, height, data = main_function(rgb)  # ALL parameters: original defaults.
        main_image = Image.fromarray(data)
        main_image.save(main_path)
        record["main"].update(status="fallback" if width is None or height is None else "ok",
                              size=list(main_image.size), output=str(main_path), parameters="original defaults")
    except Exception as error:
        record["main"].update(status="error", error=str(error))
    record["main"]["log"] = stream.getvalue()

    command = [sys.executable, str(ROOT/"realpixelart.py"), "-i", str(shared_path), "-o", str(real_path)]
    if debug_dir is not None:
        command.extend(["--debug", "--debug-dir", str(debug_dir)])
    # Real CLI invocation: no pixel-size, target-size, sampling, or grid overrides.
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
    record["realpixelart"].update(command=command, stdout=completed.stdout, stderr=completed.stderr,
                          returncode=completed.returncode)
    if completed.returncode == 0:
        with Image.open(real_path) as image:
            real_image = image.copy()
        record["realpixelart"].update(status="fallback" if "No reliable grid evidence" in completed.stderr else "ok",
                              size=list(real_image.size), output=str(real_path))
        if debug_dir is not None:
            info = json.loads((Path(debug_dir)/"info.json").read_text(encoding="utf-8"))
            record["realpixelart"].update(grid=info["grid"], debug_dir=str(debug_dir))
    else:
        record["realpixelart"].update(status="error")
    print(path.name, "main:", record["main"].get("size", "error"),
          "realpixelart:", record["realpixelart"].get("size", "error"), flush=True)
    return path.name, main_image, real_image, record


def display(image, zoom):
    image = image.resize((image.width*zoom, image.height*zoom), Image.Resampling.NEAREST).convert("RGBA")
    background = Image.new("RGBA", image.size, (255,255,255,255))
    background.alpha_composite(image)
    return background.convert("RGB")


def write_comparison(rows, path, fixed_zoom=None):
    layouts = []
    for _, left, right, _ in rows:
        images = [im for im in (left,right) if im is not None]
        width = max((im.width for im in images), default=320)
        height = max((im.height for im in images), default=160)
        zoom = fixed_zoom or max(1, min(4, 520//width, 640//height))
        layouts.append((zoom, max(540,width*zoom+24), height*zoom+85))
    column_width = max(item[1] for item in layouts)
    canvas = Image.new("RGB", (column_width*2,sum(item[2] for item in layouts)), "white")
    draw = ImageDraw.Draw(canvas)
    top = 0
    for (name,left,right,record),(zoom,_,height) in zip(rows,layouts):
        for col,(label,image) in enumerate((("perfectPixel-main",left),("RealPixelArt",right))):
            x = col*column_width
            draw.text((x+12,top+8), name+" | "+label, fill="black")
            if image is None:
                draw.text((x+12,top+30), "FAILED; see comparison.json", fill="red")
                continue
            draw.text((x+12,top+28), f"Native {image.width}x{image.height}; NEAREST display {zoom}x", fill="black")
            note = record["main"]["backend"] if col == 0 else "Original CLI; shared white RGB input; default parameters"
            draw.text((x+12,top+48), note, fill="black")
            preview = display(image,zoom)
            canvas.paste(preview,(x+(column_width-preview.width)//2,top+73))
        top += height
    canvas.save(path)
    return [item[0] for item in layouts]


def run(input_path=None, output_dir=None, debug=False):
    output_dir = ROOT/"output" if output_dir is None else Path(output_dir).resolve()
    if input_path is None:
        files = sorted(p.resolve() for p in (ROOT/"input").iterdir() if p.is_file()
                       and p.suffix.lower() in (".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"))
    else:
        files = [Path(input_path).resolve()]
    if not files:
        raise ValueError("No images in input/")
    if len({path.stem.casefold() for path in files}) != len(files):
        raise ValueError("Input basenames must be distinct; compare colliding names separately")
    output_dir.mkdir(parents=True,exist_ok=True)
    function = load_main()  # Original package selects its own backend.
    rows = [run_pair(path,output_dir/"comparison_native",function,
                     output_dir/"debug"/"comparison"/path.stem if debug else None) for path in files]
    zooms = write_comparison(rows,output_dir/"comparison.png")
    for row in rows:
        write_comparison([row],output_dir/"comparison_native"/(Path(row[0]).stem+"_8x.png"),fixed_zoom=8)
    report = {"protocol": "Same EXIF-corrected source composited onto pure white using alpha, "
                          "then passed as identical RGB pixels to main public API and RealPixelArt CLI with original defaults. "
                          "Each program detects its own grid. No pattern generation, external grid selection, "
                          "resizing of inputs, synthetic scoring, or output halo removal. Display background is white.",
              "columns": ["perfectPixel-main","RealPixelArt"],
              "display_zoom_per_row": zooms, "cases": [row[3] for row in rows]}
    (output_dir/"comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return int(any(row[3][label]["status"]=="error" for row in rows for label in ("main","realpixelart")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i","--input",type=Path,help="one real image; default: images in input/")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--debug", action="store_true", help="save RealPixelArt diagnostics under output/debug/comparison/")
    args=parser.parse_args()
    raise SystemExit(run(args.input,args.output_dir,args.debug))
