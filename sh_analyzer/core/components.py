from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .morphology import as_mask, max_inscribed_diameter_px


@dataclass(frozen=True)
class Component:
    """Связная компонента бинарной маски со свойствами региона."""

    label: int
    area_px: int
    bbox: tuple[int, int, int, int]        # x, y, w, h
    centroid: tuple[float, float]          # cx, cy в координатах исходного изображения
    contour: np.ndarray                    # (N,1,2) int32, тоже в исходных координатах
    perimeter_px: float
    solidity: float                        # contourArea / convexHullArea
    circularity: float                     # 4*pi*A / P^2
    max_width_px: float                    # диаметр максимальной вписанной окружности
    extent: float                          # area / (w*h)
    aspect_ratio: float                    # w / h


def find_components(
    mask: np.ndarray,
    min_area_px: int = 1,
    max_area_px: int | None = None,
    connectivity: int = 8,
) -> list[Component]:
    """Компоненты маски, отсортированы по убыванию площади."""
    m = as_mask(mask)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity)

    out: list[Component] = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a < min_area_px:
            continue
        if max_area_px is not None and a > max_area_px:
            continue

        crop = (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)

        ca = float(cv2.contourArea(c))
        hull = float(cv2.contourArea(cv2.convexHull(c)))
        perimeter = float(cv2.arcLength(c, True))

        solidity = min(1.0, ca / hull) if hull > 0 else 1.0
        circularity = float(4.0 * math.pi * a / (perimeter * perimeter)) if perimeter > 0 else 0.0
        width = max_inscribed_diameter_px(crop)

        contour = c.astype(np.int32) + np.array([x, y], dtype=np.int32)

        out.append(Component(
            label=i,
            area_px=a,
            bbox=(x, y, w, h),
            centroid=(float(cents[i][0]), float(cents[i][1])),
            contour=contour,
            perimeter_px=perimeter,
            solidity=round(solidity, 4),
            circularity=round(circularity, 4),
            max_width_px=round(width, 3),
            extent=round(float(a / (w * h)), 4) if w * h > 0 else 0.0,
            aspect_ratio=round(float(w / h), 4) if h > 0 else 0.0,
        ))

    out.sort(key=lambda comp: -comp.area_px)
    return out
