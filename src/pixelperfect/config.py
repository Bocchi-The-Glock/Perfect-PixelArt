"""Validated, immutable configuration shared by the CLI and Python API."""
from dataclasses import dataclass
import math
from .palette import validate_color_options


def validate_scale(scale):
    if isinstance(scale, bool) or not isinstance(scale, int) or not 1 <= scale <= 16:
        raise ValueError("scale must be an integer from 1 to 16")


@dataclass(frozen=True)
class Config:
    colors: int | None = None
    palette: str | None = None
    color_mode: str = "natural"
    scale: int = 1
    sampling: str = "robust"
    alpha_mode: str = "auto"
    local_warp: str = "auto"
    min_pixel_size: float = 2.0
    max_pixel_size: float = 64.0
    square: bool = False
    confidence_threshold: float = 0.45

    def __post_init__(self):
        validate_color_options(self.colors, self.palette, self.color_mode)
        validate_scale(self.scale)
        if self.sampling not in ("robust", "center", "median"):
            raise ValueError("sampling must be robust, center, or median")
        if self.alpha_mode not in ("auto", "binary", "coverage"):
            raise ValueError("alpha_mode must be auto, binary, or coverage")
        if self.local_warp not in ("auto", "off"):
            raise ValueError("local_warp must be auto or off")
        if not (math.isfinite(self.min_pixel_size) and math.isfinite(self.max_pixel_size)
                and 1 <= self.min_pixel_size <= self.max_pixel_size):
            raise ValueError("scale range must satisfy 1 <= min_pixel_size <= max_pixel_size")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must lie in [0,1]")
