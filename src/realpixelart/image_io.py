"""Image boundaries: EXIF orientation, canonical RGBA, and PNG-only export."""
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from .config import validate_scale


@dataclass
class ImageData:
    rgba: np.ndarray
    has_alpha: bool
    metadata: dict = field(default_factory=dict)


def _oriented_photo(image):
    """Read one photo, including the primary JPEG in an MPF/MPO container.

    MPO is a multi-picture photograph container, not an animation format. Use
    its unique declared primary image, otherwise the first image, explicitly.
    Do not change the caller's current frame. Other multi-frame formats remain
    unsupported; never silently select a GIF/APNG/TIFF frame.
    """
    frames = getattr(image, 'n_frames', 1)
    if frames != 1 and image.format != 'MPO':
        raise ValueError(f"only single-frame static images are supported; "
                         f"detected format={image.format or 'unknown'}, frames={frames}. "
                         "Export the intended photo/frame as a static PNG or JPEG.")
    selected = 0
    policy = 'single image'
    if frames > 1:
        entries = getattr(image, 'mpinfo', {}).get(0xB002, [])
        primary = [i for i, entry in enumerate(entries[:frames])
                   if entry.get('Attribute', {}).get('MPType') == 'Baseline MP Primary Image']
        selected = primary[0] if len(primary) == 1 else 0
        policy = 'declared MP primary image' if len(primary) == 1 else 'first MPO image'
    metadata = dict(format=image.format or 'PIL', frames=frames,
                    selected_frame=selected, selection=policy)
    previous = image.tell()
    try:
        if previous != selected:
            image.seek(selected)
        oriented = ImageOps.exif_transpose(image)
    finally:
        if image.tell() != previous:
            image.seek(previous)
    return oriented, metadata


def load_image(image) -> ImageData:
    """Accept a path, PIL image, or HxWx3/4 uint8 / float [0,1] array.

    Computation uses encoded sRGB, not linear radiometry. Hidden RGB is cleared
    only where alpha is zero; the visible input is otherwise left untouched.
    """
    if isinstance(image, np.ndarray):
        metadata = dict(format='array', frames=1, selected_frame=0, selection='single image')
        a = np.asarray(image)
        if a.ndim != 3 or a.shape[-1] not in (3, 4) or min(a.shape[:2]) < 1:
            raise ValueError("image array must have positive shape HxWx3 or HxWx4")
        if a.dtype == np.uint8:
            a = a.astype(np.float32)
            a /= 255.0
        elif a.dtype.kind == "f":
            if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
                raise ValueError("floating image values must be finite and in [0,1]")
            a = a.astype(np.float32, copy=True)
        else:
            raise ValueError("image array must be uint8 or floating point in [0,1]")
        has_alpha = a.shape[2] == 4
        if not has_alpha:
            a = np.concatenate((a, np.ones((*a.shape[:2], 1), np.float32)), axis=2)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                if isinstance(image, Image.Image):
                    # Copying a PIL image can discard a modified in-memory Exif
                    # object. Transpose first; exif_transpose returns a copy.
                    im, metadata = _oriented_photo(image)
                else:
                    with Image.open(image) as opened:
                        im, metadata = _oriented_photo(opened)
                if min(im.size) < 1:
                    raise ValueError("image dimensions must be positive")
                has_alpha = "A" in im.getbands() or "transparency" in im.info
                # One full float buffer; avoid full RGBA conversion, tobytes and
                # a second float buffer being alive together on large photos.
                try:
                    a = np.empty((im.height, im.width, 4), np.float32)
                    for start in range(0, im.height, 64):
                        with im.crop((0, start, im.width, min(start + 64, im.height))) as stripe:
                            with stripe.convert('RGBA') as rgba:
                                a[start:start + 64] = np.asarray(rgba)
                        a[start:start + 64] /= 255.0
                finally:
                    im.close()
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ValueError(f"cannot read valid static image: {exc}") from exc
    for start in range(0, len(a), 64):
        stripe = a[start:start + 64]
        stripe[stripe[..., 3] == 0, :3] = 0
    return ImageData(np.ascontiguousarray(a), has_alpha, metadata)


def to_pil(rgba: np.ndarray, has_alpha: bool = True) -> Image.Image:
    a = np.empty(rgba.shape, np.uint8)
    for start in range(0, len(rgba), 64):
        values = np.clip(rgba[start:start + 64], 0, 1)
        values *= 255
        np.rint(values, out=values)
        stripe = a[start:start + 64]
        stripe[:] = values
        stripe[stripe[..., 3] == 0, :3] = 0
    return Image.fromarray(a if has_alpha else a[..., :3])


def export_png(image, path, scale: int = 1) -> None:
    """Export an already recovered native image; never run grid detection again."""
    validate_scale(scale)
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("output must have a .png extension")
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def save_result(result, path, scale: int = 1, debug_dir=None, *, debug=False) -> None:
    """Save PNG; write diagnostics only when debug=True."""
    if debug_dir is not None and not debug:
        raise ValueError("debug_dir requires debug=True")
    from time import perf_counter
    started = perf_counter()
    path = Path(path)
    export_png(result.image, path, scale)
    result.timings["save_png"] = perf_counter() - started
    result.timings.pop("debug_images", None)
    result.timings["total_with_export"] = result.timings["total"] + result.timings["save_png"]
    if debug:
        from .diagnostics import write_debug
        directory = Path(debug_dir) if debug_dir is not None else path.parent / "debug" / path.stem
        write_debug(result, directory=directory, export_path=path, export_scale=scale)
