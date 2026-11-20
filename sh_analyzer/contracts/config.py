from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from ..errors import ConfigError, ScaleError


class ScaleConfig(BaseModel):
    mm_per_px: float | None = None
    dpi: float | None = 300.0
    ruler_auto: bool = True
    ruler_tolerance: float = 0.02
    ruler_strip_frac: float = 0.12


class SampleConfig(BaseModel):
    exclude_left_frac: float = 0.10
    expected_samples: int | None = None
    min_area_mm2: float = 300.0
    close_mm: float = 2.0
    edge_margin_mm: float = 1.5
    band_dark_ratio: float = 0.45
    band_open_mm: float = 2.0
    split_max_erosion_mm: float = 15.0


class SpotConfig(BaseModel):
    """Основной детектор: цветное пятно прожога/деметаллизации."""
    warmth_min: int = 8
    open_mm: float = 0.10
    merge_gap_mm: float = 0.50
    fill_internal_holes: bool = True
    min_diameter_mm: float = 0.5
    max_area_mm2: float = 300.0
    min_solidity: float = 0.30
    shape_filter_min_px: int = 60
    possibly_merged_area_mm2: float = 4.0
    possibly_merged_solidity: float = 0.55

    @property
    def min_area_mm2(self) -> float:
        return math.pi * (self.min_diameter_mm / 2.0) ** 2


class HaloConfig(BaseModel):
    """Вспомогательный детектор светлого поля. По умолчанию выключен."""
    enabled: bool = False
    delta: int = 14
    open_mm: float = 0.15
    close_mm: float = 0.30
    bg_window_mm: float = 12.0
    allow_light_only: bool = False


class FrameConfig(BaseModel):
    mode: Literal["sample", "scan"] = "sample"
    y_up: bool = True


class OutputConfig(BaseModel):
    save_overlay: bool = True
    overlay_max_side: int = 2200
    save_debug_masks: bool = False
    include_contours: bool = True
    contour_eps_mm: float = 0.15
    save_json: bool = True


class PerfConfig(BaseModel):
    resize_max_side: int | None = None
    workers: int = 2


class Config(BaseModel):
    """Все физические пороги — в миллиметрах, пересчитываются через мм/пиксель."""

    profile_name: str = "default"
    scale: ScaleConfig = Field(default_factory=ScaleConfig)
    samples: SampleConfig = Field(default_factory=SampleConfig)
    spots: SpotConfig = Field(default_factory=SpotConfig)
    halo: HaloConfig = Field(default_factory=HaloConfig)
    frame: FrameConfig = Field(default_factory=FrameConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    perf: PerfConfig = Field(default_factory=PerfConfig)

    # --- масштаб ---
    @property
    def mm_per_px(self) -> float:
        if self.scale.mm_per_px and self.scale.mm_per_px > 0:
            return float(self.scale.mm_per_px)
        if self.scale.dpi and self.scale.dpi > 0:
            return 25.4 / float(self.scale.dpi)
        raise ScaleError("Не задан масштаб: укажи scale.mm_per_px или scale.dpi")

    @property
    def scale_source(self) -> str:
        return "manual" if self.scale.mm_per_px else "dpi"

    # --- ввод/вывод профиля ---
    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Профиль не найден: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        return cls.from_yaml(path) if path else cls()

    def to_yaml(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.model_dump(), allow_unicode=True, sort_keys=False),
                     encoding="utf-8")

    def override(self, **kwargs) -> Config:
        """Точечная правка вложенных полей: override(**{'spots.warmth_min': 6})."""
        data = self.model_dump()
        for key, val in kwargs.items():
            parts = key.split(".")
            node = data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = val
        return Config.model_validate(data)