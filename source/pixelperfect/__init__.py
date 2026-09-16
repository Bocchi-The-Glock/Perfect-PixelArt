"""Public API. No file is written by pixelize()."""
from .config import Config
from .image_io import save_result
from .pipeline import PixelizeResult, pixelize

__all__ = ["Config", "PixelizeResult", "pixelize", "save_result"]
__version__ = "0.1.0"
