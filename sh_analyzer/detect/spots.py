"""Детектор зон СВ: цветовая аномалия (R-B) относительно сине-серой металлизации.

Определение зоны (по ручной разметке заказчика):
  компактное пятно прожога/деметаллизации, граница по видимому изменению цвета.
  Лучи-складки и окружающее просветление в площадь НЕ входят.
  Кластер долек ближе merge_gap_mm = ОДНА зона.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..contracts.config import Config
from ..core.components import find_components
from ..core.geometry import area_px_to_mm2, mm_to_px, px_to_mm
from ..core.morphology import as_mask, close_disk, fill_holes, open_disk


@dataclass
class SpotRegion:
    center_px: tuple[float, float]
    area_px: int
    bbox: tuple[int, int, int, int]
    contour: np.ndarray
    circularity: float
    solidity: float
    max_width_px: float
    warm_mean: float
    n_events: int = 1
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)


def warmth(bgr: np.ndarray) -> np.ndarray:
    """R - B. Плёнка ~ -30, пятно прожога +20..+50, царапины/текст ~ 0."""
    b, _, r = cv2.split(bgr.astype(np.int16))
    return r - b


def spot_mask(bgr: np.ndarray, active: np.ndarray, cfg: Config, s: float
              ) -> tuple[np.ndarray, np.ndarray]:
    warm = warmth(bgr)
    m = ((warm > cfg.spots.warmth_min) & active).astype(np.uint8) * 255
    m = open_disk(m, mm_to_px(cfg.spots.open_mm, s))               # одиночный шум
    m = close_disk(m, mm_to_px(cfg.spots.merge_gap_mm / 2.0, s))   # склейка долек
    if cfg.spots.fill_internal_holes:
        m = fill_holes(m)                                          # блики внутри пятна
    m[~active] = 0                                                 # не вылезать за область
    return m, warm


def detect_spots(bgr: np.ndarray, active: np.ndarray, cfg: Config, s: float
                 ) -> tuple[list[SpotRegion], dict]:
    m, warm = spot_mask(bgr, active, cfg, s)

    min_px = max(1, int(round(cfg.spots.min_area_mm2 / s ** 2)))
    max_px = int(round(cfg.spots.max_area_mm2 / s ** 2))
    comps = find_components(m, min_area_px=min_px, max_area_px=max_px)

    out: list[SpotRegion] = []
    for comp in comps:
        # форма на мелких пятнах статистически ненадёжна — фильтруем только крупные
        if (comp.area_px >= cfg.spots.shape_filter_min_px
                and comp.solidity < cfg.spots.min_solidity):
            continue

        x, y, w, h = comp.bbox
        local = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(local, [comp.contour - np.array([x, y], dtype=np.int32)], -1, 255, -1)
        wm = float(warm[y:y + h, x:x + w][local > 0].mean()) if local.any() else 0.0

        area_mm2 = area_px_to_mm2(comp.area_px, s)
        width_mm = px_to_mm(comp.max_width_px, s)

        flags: list[str] = []
        if (area_mm2 > cfg.spots.possibly_merged_area_mm2
                and comp.solidity < cfg.spots.possibly_merged_solidity):
            flags.append("possibly_merged")   # на ручной просмотр, автоматически НЕ делим

        conf = 0.6
        conf += 0.15 if wm > cfg.spots.warmth_min + 10 else 0.0
        conf += 0.10 if comp.circularity > 0.5 else 0.0
        conf += 0.10 if width_mm >= 0.8 else 0.0
        conf -= 0.20 if flags else 0.0

        out.append(SpotRegion(
            center_px=comp.centroid,
            area_px=comp.area_px,
            bbox=comp.bbox,
            contour=comp.contour,
            circularity=comp.circularity,
            solidity=comp.solidity,
            max_width_px=comp.max_width_px,
            warm_mean=round(wm, 1),
            n_events=1,
            confidence=round(min(0.95, max(0.05, conf)), 3),
            flags=flags,
        ))

    # опциональный добор СВ без цветного пятна (по умолчанию выключено)
    if cfg.spots.allow_light_only:
        from .halo import detect_halo
        halos, _ = detect_halo(bgr, active, cfg, s)
        for hz in halos:
            inside = any(
                cv2.pointPolygonTest(z.contour, hz.centroid, False) >= 0 for z in out
            )
            if inside:
                continue
            out.append(SpotRegion(
                center_px=hz.centroid, area_px=hz.area_px, bbox=hz.bbox,
                contour=hz.contour, circularity=hz.circularity, solidity=hz.solidity,
                max_width_px=hz.max_width_px, warm_mean=0.0, n_events=1,
                confidence=0.35, flags=["light_only"],
            ))

    out.sort(key=lambda z: (z.center_px[1], z.center_px[0]))
    debug = {
        "spot": m,
        "warm": np.clip(warm + 60, 0, 255).astype(np.uint8),
        "active": as_mask(active),
    }
    return out, debug
