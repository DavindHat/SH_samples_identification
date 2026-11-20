from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


def mm_to_px(mm: float, s: float) -> int:
    """мм -> целое число пикселей (минимум 1). s = мм/пиксель."""
    return max(1, int(round(mm / s)))


def px_to_mm(px: float, s: float) -> float:
    return float(px) * s


def area_px_to_mm2(area_px: float, s: float) -> float:
    return float(area_px) * s * s


def equivalent_diameter_mm(area_mm2: float) -> float:
    return 2.0 * math.sqrt(max(0.0, area_mm2) / math.pi)


@dataclass(frozen=True)
class Basis:
    """Локальная система координат образца. e_y смотрит 'вверх' на изображении."""
    centroid: np.ndarray      # (cx, cy) в пикселях
    e_x: np.ndarray           # единичный вектор X (вправо)
    e_y: np.ndarray           # единичный вектор Y (вверх по изображению)
    tilt_deg: float


def sample_basis(mask: np.ndarray, mode: str = "sample") -> Basis:
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        raise ValueError("Пустая маска образца")
    c = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=float)

    if mode == "scan":
        e_y = np.array([0.0, -1.0])
    else:
        theta = 0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])
        e_y = np.array([math.cos(theta), math.sin(theta)])
        if e_y[1] > 0:
            e_y = -e_y
    e_x = np.array([-e_y[1], e_y[0]])
    tilt = math.degrees(math.atan2(e_y[0], -e_y[1]))
    return Basis(c, e_x, e_y, float(tilt))


def basis_from_tilt(centroid_px: tuple[float, float], tilt_deg: float) -> Basis:
    """Восстановление базиса из сохранённых центроида и наклона (для валидатора)."""
    t = math.radians(tilt_deg)
    e_y = np.array([math.sin(t), -math.cos(t)])
    e_x = np.array([math.cos(t), math.sin(t)])
    return Basis(np.array(centroid_px, dtype=float), e_x, e_y, float(tilt_deg))


def to_frame(points_px, basis: Basis, s: float) -> np.ndarray:
    """Пиксели скана -> мм в рамке образца (Y вверх)."""
    d = np.atleast_2d(np.asarray(points_px, dtype=float)) - basis.centroid
    return np.column_stack([d @ basis.e_x * s, d @ basis.e_y * s])


def from_frame(points_mm, basis: Basis, s: float) -> np.ndarray:
    """мм в рамке образца -> пиксели скана."""
    p = np.atleast_2d(np.asarray(points_mm, dtype=float))
    return (basis.centroid
            + np.outer(p[:, 0] / s, basis.e_x)
            + np.outer(p[:, 1] / s, basis.e_y))


def polygon_area_px(contour: np.ndarray) -> float:
    return float(cv2.contourArea(np.asarray(contour, np.int32)))


def contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(np.asarray(contour, np.int32))
    if m["m00"] == 0:
        pts = np.asarray(contour).reshape(-1, 2).mean(axis=0)
        return float(pts[0]), float(pts[1])
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])