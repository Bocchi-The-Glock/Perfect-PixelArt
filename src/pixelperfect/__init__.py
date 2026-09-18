"""Public API. No file is written by pixelize()."""
from .config import Config
from .image_io import export_png, save_result
from .pipeline import PixelizeResult, pixelize
from .palette import ColorResult, process_colors, palette_catalog

__all__ = ["Config", "PixelizeResult", "pixelize", "save_result", "export_png",
           "ColorResult", "process_colors", "palette_catalog"]
__version__ = "0.3.0"
