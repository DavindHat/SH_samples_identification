"""Светлое поле (просветление) вокруг прожога. Только диагностика/добор.

В площадь зоны СВ НЕ входит — по разметке заказчика площадь считается
по цветному пятну (detect/spots.py).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..contracts.config import Config
from ..core.components import Component, find_components
from ..core.geometry import mm_to_px
from ..core.morphology import close_disk, open_disk


def estimate_background(gray: np.ndarray, valid: np.ndarray,
                        window_px: int, downscale: int = 8) -> np.ndarray:
    """Оценка спокойного фона плёнки: медиана на уменьшенной копии."""
    g = gray.copy()
    med = int(np.median(gray[valid])) if valid.any() else 0
    g[~valid] = med
    f = max(1, int(downscale))
    small = cv2.resize(g, None, fx=1.0 / f, fy=1.0 / f, interpolation=cv2.INTER_AREA)
    k = max(3, (int(window_px) // f) | 1)          # обязательно нечётное
    small = cv2.medianBlur(small, k)
    return cv2.resize(small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def detect_halo(bgr: np.ndarray, active: np.ndarray, cfg: Config, s: float
                ) -> tuple[list[Component], dict]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg = estimate_background(gray, active,
                             mm_to_px(cfg.halo.bg_window_mm, s), cfg.perf.bg_downscale)
    diff = gray.astype(np.int16) - bg.astype(np.int16)

    m = ((diff > cfg.halo.delta) & active).astype(np.uint8) * 255
    m = open_disk(m, mm_to_px(cfg.halo.open_mm, s))     # убрать тонкие складки
    m = close_disk(m, mm_to_px(cfg.halo.close_mm, s))
    m[~active] = 0

    comps = find_components(
        m,
        min_area_px=max(1, int(round(cfg.halo.min_area_mm2 / s ** 2))),
        max_area_px=int(round(cfg.halo.max_area_mm2 / s ** 2)),
    )
    debug = {"halo": m, "bg": bg, "diff": np.clip(diff + 128, 0, 255).astype(np.uint8)}
    return comps, debug
