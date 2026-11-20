from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from pydantic import BaseModel, Field

from ..version import SCHEMA_VERSION, __version__


class Zone(BaseModel):
    """Одна зона СВ. Площадь = площадь цветного пятна / n_events."""

    id: int
    x_mm: float
    y_mm: float
    center_px: tuple[int, int]

    area_mm2: float
    area_px: int                       # площадь всего пятна до деления
    n_events: int = 1                  # >1 только при ручном указании слипшихся
    d_eq_mm: float
    circularity: float | None = None
    max_width_mm: float | None = None
    solidity: float | None = None
    warm_mean: float | None = None

    bbox_mm: tuple[float, float, float, float] | None = None
    bbox_px: tuple[int, int, int, int] | None = None
    contour_mm: list[tuple[float, float]] | None = None

    confidence: float = 0.0
    flags: list[str] = Field(default_factory=list)


class Sample(BaseModel):
    index: int
    label: str | None = None
    sample_id: str | None = None

    centroid_px: tuple[int, int]
    tilt_deg: float
    area_mm2: float
    active_area_mm2: float
    bbox_px: tuple[int, int, int, int]

    n_zones: int = 0
    total_zone_area_mm2: float = 0.0
    zone_area_fraction: float = 0.0
    zones: list[Zone] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ScanResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    analyzer_version: str = __version__

    file: str
    sha256: str = ""
    image_size_px: tuple[int, int] = (0, 0)
    mm_per_px: float = 0.0
    scale_source: str = "dpi"

    session_id: str | None = None
    acquired_at: str | None = None
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    device: dict = Field(default_factory=dict)
    scan_profile: dict = Field(default_factory=dict)

    n_samples: int = 0
    n_zones: int = 0
    samples: list[Sample] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)

    def iter_zones(self) -> Iterator[tuple[Sample, Zone]]:
        for sm in self.samples:
            for z in sm.zones:
                yield sm, z

    def save_json(self, path: str) -> None:
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")