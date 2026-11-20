from __future__ import annotations

import cv2
import numpy as np


def disk(radius_px: int) -> np.ndarray:
    r = max(1, int(radius_px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def open_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(radius_px))


def close_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(radius_px))


def erode(mask: np.ndarray, radius_px: int) -> np.ndarray:
    return cv2.erode(mask, disk(radius_px))


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    return cv2.dilate(mask, disk(radius_px))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Заливает внутренние дырки (блёстки, просветы внутри пятна)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, -1)
    return out


def remove_small(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            out[lab == i] = 255
    return out


def as_mask(condition: np.ndarray) -> np.ndarray:
    return condition.astype(np.uint8) * 255