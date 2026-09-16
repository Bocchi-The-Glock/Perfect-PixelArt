"""Validated, immutable configuration shared by the CLI and Python API."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Config:
    pixel_size: float | tuple[float, float] | None = None
    target_size: tuple[int, int] | None = None
    colors: int | None = None
    scale: int = 1
    sampling: str = "robust"
    local_warp: str = "auto"
    min_pixel_size: float = 2.0
    max_pixel_size: float = 64.0
    square: bool = False
    max_axis_candidates: int = 6
    max_candidates: int = 4
    max_samples: int = 128
    complexity_weight: float = 0.045
    structure_weight: float = 0.035
    grid_weight: float = 0.02
    confidence_threshold: float = 0.45

    def __post_init__(self):
        if self.pixel_size is not None and self.target_size is not None:
            raise ValueError("pixel_size and target_size are mutually exclusive")
        if self.pixel_size is not None:
            sizes = ((self.pixel_size, self.pixel_size) if isinstance(self.pixel_size, (int, float))
                     else tuple(self.pixel_size))
            if len(sizes) != 2 or not all(math.isfinite(float(s)) and float(s) >= 1 for s in sizes):
                raise ValueError("pixel_size must contain two finite spacings >= 1")
            object.__setattr__(self, "pixel_size", tuple(float(s) for s in sizes))
            if self.square and abs(sizes[0] - sizes[1]) > 1e-8:
                raise ValueError("square mode requires equal pixel spacings")
        if self.target_size is not None:
            sizes = tuple(self.target_size)
            if len(sizes) != 2 or any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in sizes):
                raise ValueError("target_size must be a positive integer (width, height)")
            object.__setattr__(self, "target_size", sizes)
        if self.colors is not None and (isinstance(self.colors, bool) or not isinstance(self.colors, int) or not 1 <= self.colors <= 256):
            raise ValueError("colors must be an integer from 1 to 256")
        if isinstance(self.scale, bool) or not isinstance(self.scale, int) or not 1 <= self.scale <= 64:
            raise ValueError("scale must be an integer from 1 to 64")
        if self.sampling not in ("robust", "center", "median"):
            raise ValueError("sampling must be robust, center, or median")
        if self.local_warp not in ("auto", "off"):
            raise ValueError("local_warp must be auto or off")
        if not (math.isfinite(self.min_pixel_size) and math.isfinite(self.max_pixel_size)
                and 1 <= self.min_pixel_size <= self.max_pixel_size):
            raise ValueError("scale range must satisfy 1 <= min_pixel_size <= max_pixel_size")
        for name, lo, hi in (("max_axis_candidates", 1, 6), ("max_candidates", 1, 8), ("max_samples", 16, 512)):
            value = getattr(self, name)
            if not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer between {lo} and {hi}")
        for name in ("complexity_weight", "structure_weight", "grid_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must lie in [0,1]")
