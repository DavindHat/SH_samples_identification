from __future__ import annotations

import cv2
import numpy as np


def as_mask(mask: np.ndarray) -> np.ndarray:
    """bool / 0-1 / 0-255 -> uint8 0/255."""
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8) * 255
    m = mask
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = m.astype(np.uint8, copy=False)
    if m.max() <= 1:
        m = m * 255
    return m


def disk(radius_px: int) -> np.ndarray:
    r = max(1, int(radius_px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def open_disk(mask: np.ndarray, r: int) -> np.ndarray:
    return cv2.morphologyEx(as_mask(mask), cv2.MORPH_OPEN, disk(r))


def close_disk(mask: np.ndarray, r: int) -> np.ndarray:
    return cv2.morphologyEx(as_mask(mask), cv2.MORPH_CLOSE, disk(r))


def erode_disk(mask: np.ndarray, r: int) -> np.ndarray:
    return cv2.erode(as_mask(mask), disk(r))


def dilate_disk(mask: np.ndarray, r: int) -> np.ndarray:
    return cv2.dilate(as_mask(mask), disk(r))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Заполняет все внутренние дырки (блики, надписи, зоны СВ)."""
    m = as_mask(mask)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 255, -1)
    return out


def max_inscribed_diameter_px(mask: np.ndarray) -> float:
    """Диаметр максимальной вписанной окружности, px.

    Нулевой бордюр обязателен: без него distanceTransform на crop,
    целиком заполненном foreground, возвращает мусор (overflow).
    """
    if mask.size == 0 or not mask.any():
        return 0.0
    padded = cv2.copyMakeBorder(as_mask(mask), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    return float(dist.max()) * 2.0
