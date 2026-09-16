"""Image boundaries: EXIF orientation, canonical RGBA, and PNG-only export."""
from dataclasses import dataclass
from pathlib import Path
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass
class ImageData:
    rgba: np.ndarray
    has_alpha: bool


def load_image(image) -> ImageData:
    """Accept a path, PIL image, or HxWx3/4 uint8 / float [0,1] array.

    Computation uses encoded sRGB, not linear radiometry. Hidden RGB is cleared
    only where alpha is zero; the visible input is otherwise left untouched.
    """
    if isinstance(image, np.ndarray):
        a = np.asarray(image)
        if a.ndim != 3 or a.shape[-1] not in (3, 4) or min(a.shape[:2]) < 1:
            raise ValueError("image array must have positive shape HxWx3 or HxWx4")
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
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
                    if getattr(image, "n_frames", 1) != 1:
                        raise ValueError("only single-frame static images are supported")
                    # Copying a PIL image can discard a modified in-memory Exif
                    # object. Transpose first; exif_transpose returns a copy.
                    im = ImageOps.exif_transpose(image)
                else:
                    with Image.open(image) as opened:
                        if getattr(opened, "n_frames", 1) != 1:
                            raise ValueError("only single-frame static images are supported")
                        opened.load()
                        im = opened.copy()
                im = ImageOps.exif_transpose(im)
                if min(im.size) < 1:
                    raise ValueError("image dimensions must be positive")
                has_alpha = "A" in im.getbands() or "transparency" in im.info
                a = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ValueError(f"cannot read valid static image: {exc}") from exc
    a[a[..., 3] == 0, :3] = 0
    return ImageData(np.ascontiguousarray(a), has_alpha)


def to_pil(rgba: np.ndarray, has_alpha: bool = True) -> Image.Image:
    a = np.rint(np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    a[a[..., 3] == 0, :3] = 0
    return Image.fromarray(a if has_alpha else a[..., :3])


def save_result(result, path, scale: int = 1) -> None:
    """Save native resolution unless explicit nearest-neighbor scale is given."""
    if isinstance(scale, bool) or not isinstance(scale, int) or not 1 <= scale <= 64:
        raise ValueError("scale must be an integer from 1 to 64")
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("output must have a .png extension")
    im = result.image
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG")
