"""Автокалибровка мм/пиксель по линейке в левой полосе скана.

Логика: находим тело линейки -> компенсируем небольшой наклон ->
профиль яркости вдоль кромки -> пики = риски -> шаг в пикселях.
Единица риска (0.5/1/2/5/10 мм) выбирается по близости к априорному DPI,
поэтому линейка УТОЧНЯЕТ масштаб, а не задаёт его вслепую.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import find_peaks

from ..contracts.config import Config
from ..core.components import find_components
from ..core.morphology import close_disk


@dataclass
class RulerScale:
    mm_per_px: float
    n_ticks: int
    spacing_px: float
    spacing_cv: float
    angle_deg: float
    unit_mm: float
    deviation: float          # относительное отклонение от априорного DPI
    confidence: float
    band: str


def _profile(gray_rot: np.ndarray, body_rot: np.ndarray, band_frac: float):
    ys, xs = np.nonzero(body_rot)
    if ys.size < 100:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bw = max(3, int(round(band_frac * (x1 - x0))))
    out = []
    for name, xa, xb in (("left", x0, x0 + bw), ("right", x1 - bw, x1)):
        if xb - xa < 2 or y1 - y0 < 20:
            continue
        prof = gray_rot[y0:y1, xa:xb].mean(axis=1).astype(np.float64)
        score = float(np.std(np.diff(prof)))
        out.append((score, name, prof, (xa, xb), y0))
    return out or None


def estimate_scale_from_ruler(bgr: np.ndarray, cfg: Config, prior_mm_per_px: float
                              ) -> tuple[RulerScale | None, dict]:
    H, W = bgr.shape[:2]
    strip_w = max(20, int(W * cfg.ruler.strip_frac))
    strip = bgr[:, :strip_w]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = close_disk(dark, 3)
    comps = find_components(dark, min_area_px=max(50, int(0.005 * dark.size)))
    if not comps:
        return None, {"reason": "тело линейки не найдено"}

    body = comps[0]
    body_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(body_mask, [body.contour], -1, 255, -1)
    cx, cy = body.centroid

    best = None
    angles = np.arange(-cfg.ruler.max_tilt_deg,
                       cfg.ruler.max_tilt_deg + 1e-9, cfg.ruler.tilt_step_deg)
    for a in angles:
        M = cv2.getRotationMatrix2D((cx, cy), float(a), 1.0)
        gr = cv2.warpAffine(gray, M, (strip_w, H), flags=cv2.INTER_LINEAR)
        bm = cv2.warpAffine(body_mask, M, (strip_w, H), flags=cv2.INTER_NEAREST)
        cand = _profile(gr, bm, cfg.ruler.edge_band_frac)
        if not cand:
            continue
        for score, name, prof, band_x, y0 in cand:
            if best is None or score > best[0]:
                best = (score, float(a), name, prof, band_x, y0, gr)
    if best is None:
        return None, {"reason": "не удалось построить профиль кромок"}

    score, angle, band, prof, band_x, y0, gr = best
    span = float(prof.max() - prof.min())
    if span < 10:
        return None, {"reason": "слишком слабый контраст рисок", "span": span}

    expected_px = 1.0 / prior_mm_per_px
    dist = max(2, int(round(0.6 * expected_px)))
    peaks, _ = find_peaks(prof, distance=dist, prominence=0.15 * span)
    if peaks.size < cfg.ruler.min_ticks:
        return None, {"reason": f"мало рисок: {peaks.size}", "peaks": peaks, "angle": angle}

    d = np.diff(peaks).astype(np.float64)
    med = float(np.median(d))
    steps = np.maximum(1, np.round(d / med)).astype(int)     # учёт пропущенных рисок
    spacing = float((peaks[-1] - peaks[0]) / steps.sum())
    single = d[steps == 1]
    cv_val = float(np.std(single) / med) if single.size > 2 else 0.0
    if cv_val > cfg.ruler.max_spacing_cv:
        return None, {"reason": f"нерегулярные риски, CV={cv_val:.3f}", "cv": cv_val}

    best_unit, best_s, best_dev = None, None, 1e9
    for u in cfg.ruler.tick_units_mm:
        s_cand = float(u) / spacing
        dev = abs(s_cand / prior_mm_per_px - 1.0)
        if dev < best_dev:
            best_unit, best_s, best_dev = float(u), s_cand, dev
    if best_dev > 0.15:
        return None, {"reason": f"шаг не согласуется с DPI (откл {best_dev:.1%})",
                      "spacing_px": spacing}

    conf = 0.5 + (0.3 if peaks.size >= 50 else 0.0) + (0.15 if cv_val < 0.10 else 0.0)
    res = RulerScale(
        mm_per_px=best_s, n_ticks=int(peaks.size), spacing_px=round(spacing, 4),
        spacing_cv=round(cv_val, 4), angle_deg=angle, unit_mm=best_unit,
        deviation=round(best_dev, 5), confidence=round(min(0.95, conf), 3), band=band,
    )
    info = {"strip_rot": gr, "peaks": peaks + y0, "band_x": band_x,
            "angle": angle, "profile": prof, "y0": y0}
    return res, info
