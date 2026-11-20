from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ScaleConfig(BaseModel):
    dpi: float = 300.0
    mm_per_px: float | None = None
    auto_ruler: bool = True
    ruler_max_deviation: float = 0.02


class SampleConfig(BaseModel):
    exclude_left_frac: float = 0.10
    expected: int | None = None
    min_area_mm2: float = 300.0
    close_mm: float = 2.0
    edge_margin_mm: float = 1.5
    band_dark_ratio: float = 0.45
    band_open_mm: float = 2.0
    band_dilate_mm: float = 0.5
    split_max_erosion_mm: float = 15.0
    split_step_mm: float = 0.5
    exclude_top_fraction: float = 0.15
    exclude_bottom_fraction: float = 0.15


class SpotConfig(BaseModel):
    warmth_min: int = 10
    open_mm: float = 0.10
    merge_gap_mm: float = 1.00
    fill_internal_holes: bool = True
    min_diameter_mm: float = 0.5
    max_area_mm2: float = 300.0
    min_solidity: float = 0.30
    shape_filter_min_px: int = 60
    possibly_merged_area_mm2: float = 4.0
    possibly_merged_solidity: float = 0.55
    allow_light_only: bool = False

    @property
    def min_area_mm2(self) -> float:
        return math.pi * (self.min_diameter_mm / 2.0) ** 2


class HaloConfig(BaseModel):
    enabled: bool = False
    delta: int = 14
    open_mm: float = 0.15
    close_mm: float = 0.30
    bg_window_mm: float = 12.0
    min_area_mm2: float = 0.2
    max_area_mm2: float = 300.0


class RulerConfig(BaseModel):
    strip_frac: float = 0.12
    max_tilt_deg: float = 6.0
    tilt_step_deg: float = 0.5
    edge_band_frac: float = 0.28
    min_ticks: int = 25
    max_spacing_cv: float = 0.25
    tick_units_mm: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0)


class FrameConfig(BaseModel):
    mode: Literal["sample", "scan"] = "sample"
    include_contours: bool = True
    contour_eps_mm: float = 0.15


class OutputConfig(BaseModel):
    save_overlay: bool = True
    overlay_max_side: int = 2200
    save_json: bool = True
    save_debug_masks: bool = True
    save_csv: bool = True
    save_xlsx: bool = True


class PerfConfig(BaseModel):
    resize_max_side: int | None = None
    bg_downscale: int = 8
    workers: int = 2


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scale: ScaleConfig = Field(default_factory=ScaleConfig)
    samples: SampleConfig = Field(default_factory=SampleConfig)
    spots: SpotConfig = Field(default_factory=SpotConfig)
    halo: HaloConfig = Field(default_factory=HaloConfig)
    ruler: RulerConfig = Field(default_factory=RulerConfig)
    frame: FrameConfig = Field(default_factory=FrameConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    perf: PerfConfig = Field(default_factory=PerfConfig)

    @property
    def mm_per_px(self) -> float:
        return float(self.scale.mm_per_px or 25.4 / self.scale.dpi)

    @property
    def frame_mode(self) -> str:
        return self.frame.mode

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


SamplesConfig = SampleConfig
SpotsConfig = SpotConfig
