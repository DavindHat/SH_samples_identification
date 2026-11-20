from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .morphology import max_inscribed_diameter_px

@dataclass
class Component:
    """Связная компонента маски со свойствами региона. Контур — в координатах скана."""
    label: int
    area_px: int
    bbox: tuple[int, int, int, int]        # x, y, w, h
    centroid: tuple[float, float]
    crop: np.ndarray                       # uint8 0/1, вырез по bbox
    contour: np.ndarray                    # (N,1,2) int32, абсолютные координаты
    perimeter_px: float
    solidity: float
    circularity: float
    max_width_px: float


def find_components(mask: np.ndarray,
                    min_area_px: int = 1,
                    max_area_px: int | None = None,
                    connectivity: int = 8,
                    with_shape: bool = True) -> list[Component]:
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity)
    out: list[Component] = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area_px:
            continue
        if max_area_px is not None and a > max_area_px:
            continue
        x, y, w, h = (int(v) for v in stats[i, :4])
        crop = (lab[y:y + h, x:x + w] == i).astype(np.uint8)

        if with_shape:
            cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)
            per = float(cv2.arcLength(c, True))
            hull = float(cv2.contourArea(cv2.convexHull(c)))
            solidity = a / hull if hull > 0 else 0.0
            width = max_inscribed_diameter_px(crop)
            width = float(2 * cv2.distanceTransform(crop, cv2.DIST_L2, 3).max())
            contour = c.astype(np.int32) + np.array([x, y], np.int32)
        else:
            per, solidity, circ, width = 0.0, 1.0, 0.0, 0.0
            contour = np.zeros((0, 1, 2), np.int32)

        out.append(Component(
            label=i, area_px=a, bbox=(x, y, w, h),
            centroid=(float(cents[i][0]), float(cents[i][1])),
            crop=crop, contour=contour, perimeter_px=per,
            solidity=solidity, circularity=circ, max_width_px=width))
    return out


def reading_order(masks: list[np.ndarray], height: int,
                  row_tol_frac: float = 0.20) -> list[np.ndarray]:
    """Сортировка 'как читаем': строки сверху вниз, внутри строки слева направо."""
    items = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        items.append((float(ys.mean()), float(xs.mean()), m))
    items.sort(key=lambda t: t[0])

    rows: list[list] = []
    cur: list = []
    for it in items:
        if cur and it[0] - cur[-1][0] > row_tol_frac * height:
            rows.append(cur)
            cur = []
        cur.append(it)
    if cur:
        rows.append(cur)

    out = []
    for row in rows:
        out.extend(t[2] for t in sorted(row, key=lambda t: t[1]))
    return out


def label_map(components: list[Component], shape: tuple[int, int]) -> np.ndarray:
    """Карта меток компонент (uint16) — для привязки точек к регионам."""
    lm = np.zeros(shape, np.uint16)
    for k, comp in enumerate(components, 1):
        x, y, w, h = comp.bbox
        sub = lm[y:y + h, x:x + w]
        sub[comp.crop > 0] = k
    return lm