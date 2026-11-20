from __future__ import annotations

import math

import cv2
import numpy as np

from ..contracts.result import Zone
from ..core.geometry import area_px_to_mm2, mm_to_px, to_frame, px_to_mm
from ..detect.spots import SpotRegion


def spots_to_zones(
    spots: list[SpotRegion],
    centroid_px: tuple[float, float],
    e_short: np.ndarray,
    e_long: np.ndarray,
    mm_per_px: float,
    include_contours: bool,
    contour_eps_mm: float,
) -> list[Zone]:
    zones: list[Zone] = []
    for i, z in enumerate(spots, 1):
        pts_mm = to_frame([z.center_px], centroid_px, e_short, e_long, mm_per_px)[0]
        x_mm, y_mm = float(pts_mm[0]), float(pts_mm[1])

        area_mm2 = area_px_to_mm2(z.area_px, mm_per_px)
        d_eq_mm = 2.0 * math.sqrt(max(0.0, area_mm2) / math.pi)
        max_width_mm = px_to_mm(z.max_width_px, mm_per_px)

        bbox_mm = None
        bbox_px = z.bbox
        contour_mm = None
        contour_px = None

        if include_contours and z.contour is not None and len(z.contour) > 2:
            # Сохраняем контур в пикселях (упрощённый)
            eps = mm_to_px(contour_eps_mm, mm_per_px)
            approx = cv2.approxPolyDP(z.contour, eps, True).reshape(-1, 2)
            contour_px = [(int(p[0]), int(p[1])) for p in approx]
            
            # Переводим в мм
            pts_approx_mm = to_frame(approx, centroid_px, e_short, e_long, mm_per_px)
            contour_mm = [(round(float(p[0]), 3), round(float(p[1]), 3)) for p in pts_approx_mm]

            xs, ys = pts_approx_mm[:, 0], pts_approx_mm[:, 1]
            bbox_mm = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

        zones.append(Zone(
            id=i,
            x_mm=round(x_mm, 3),
            y_mm=round(y_mm, 3),
            center_px=(int(round(z.center_px[0])), int(round(z.center_px[1]))),
            area_mm2=round(area_mm2, 4),
            area_px=z.area_px,
            n_events=z.n_events,
            d_eq_mm=round(d_eq_mm, 3),
            circularity=round(z.circularity, 3),
            max_width_mm=round(max_width_mm, 3),
            solidity=round(z.solidity, 3),
            warm_mean=z.warm_mean,
            bbox_mm=bbox_mm,
            bbox_px=bbox_px,
            contour_mm=contour_mm,
            contour_px=contour_px,  # НОВОЕ
            confidence=z.confidence,
            flags=z.flags,
        ))
    return zones
