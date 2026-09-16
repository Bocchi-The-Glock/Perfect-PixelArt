"""Command line interface; all diagnostic messages go to stderr."""
import argparse
from pathlib import Path
import sys
from .config import Config
from .pipeline import pixelize
from .image_io import save_result


def _spacing(text):
    try:
        parts = text.lower().split("x")
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return tuple(map(float, parts))
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("pixel size must be a number or SXxSY, e.g. 8 or 8x9")


def _dimensions(text):
    try:
        parts = tuple(map(int, text.lower().split("x")))
        if len(parts) == 2 and min(parts) > 0:
            return parts
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("target size must be positive WxH, e.g. 64x64")


def parser():
    p = argparse.ArgumentParser(description="Restore pseudo pixel art to native low-resolution PNG.")
    p.add_argument("-i", "--input", required=True, type=Path)
    p.add_argument("-o", "--output", required=True, type=Path)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--pixel-size", type=_spacing, help="source cell spacing; phase remains optimized")
    modes.add_argument("--target-size", type=_dimensions, help="exact native output WxH (preserve aspect ratio)")
    p.add_argument("--colors", type=int)
    p.add_argument("--scale", type=int, default=1, help="integer nearest-neighbor export multiplier")
    p.add_argument("--sampling", choices=("robust", "center", "median"), default="robust")
    p.add_argument("--local-warp", choices=("auto", "off"), default="auto")
    p.add_argument("--min-pixel-size", type=float, default=2)
    p.add_argument("--max-pixel-size", type=float, default=64)
    p.add_argument("--square", action="store_true", help="require equal nominal x/y spacing")
    p.add_argument("--debug-dir", type=Path)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        config = Config(pixel_size=args.pixel_size, target_size=args.target_size, colors=args.colors,
                        scale=args.scale, sampling=args.sampling, local_warp=args.local_warp,
                        min_pixel_size=args.min_pixel_size, max_pixel_size=args.max_pixel_size,
                        square=args.square)
        if args.output.suffix.lower() != ".png":
            raise ValueError("output must have a .png extension")
        if args.input.resolve() == args.output.resolve():
            raise ValueError("input and output paths must differ")
        result = pixelize(args.input, config)
        save_result(result, args.output, args.scale)
        if args.debug_dir is not None:
            from .diagnostics import write_debug
            write_debug(result, args.input, args.debug_dir)
        for message in result.diagnostics["warnings"]:
            print(f"pixelperfect: {message}", file=sys.stderr)
        if args.verbose:
            print(f"native={result.image.width}x{result.image.height}; heuristic confidence={result.confidence:.3f}; "
                  f"export scale={args.scale}", file=sys.stderr)
            for name, seconds in result.timings.items():
                print(f"  {name}: {seconds:.4f} s", file=sys.stderr)
        return 0
    except (ValueError, TypeError, OSError) as exc:
        print(f"pixelperfect: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
