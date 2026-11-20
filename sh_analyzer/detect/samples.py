from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from ..contracts.config import Config
from ..core.components import find_components
from ..core.geometry import mm_to_px, reading_order, sample_basis
from ..core.morphology import close_disk, erode_disk, fill_holes


@dataclass
class SampleRegion:
    mask: np.ndarray                       # uint8 0/255, размер всего скана
    bbox: tuple[int, int, int, int]        # x, y, w, h
    centroid: tuple[float, float]
    e_short: np.ndarray
    e_long: np.ndarray
    tilt_deg: float
    area_px: int
    flags: list[str] = field(default_factory=list)


def _split_merged(mask: np.ndarray, k: int, cfg: Config, s: float) -> list[np.ndarray]:
    """Эрозия до k маркеров -> разбиение Вороного внутри маски."""
    step = mm_to_px(cfg.samples.split_step_mm, s)
    limit = mm_to_px(cfg.samples.split_max_erosion_mm, s)
    min_marker = max(10, int(0.02 * float((mask > 0).sum()) / max(1, k)))

    markers = None
    for r in range(step, limit + 1, step):
        er = erode_disk(mask, r)
        comps = find_components(er, min_area_px=min_marker)
        if len(comps) >= k:
            markers = np.zeros(mask.shape, dtype=np.int32)
            for j, comp in enumerate(comps[:k], start=1):
                x, y, w, h = comp.bbox
                view = markers[y:y + h, x:x + w]
                view[er[y:y + h, x:x + w] > 0] = j
            break

    if markers is None:
        return [mask]

    _, (iy, ix) = distance_transform_edt(markers == 0, return_indices=True)
    assign = markers[iy, ix]
    return [((assign == j) & (mask > 0)).astype(np.uint8) * 255 for j in range(1, k + 1)]


def detect_samples(bgr: np.ndarray, cfg: Config, s: float
                   ) -> tuple[list[SampleRegion], list[str]]:
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    dark[:, : int(W * cfg.samples.exclude_left_frac)] = 0      # полоса линейки
    dark = close_disk(dark, mm_to_px(cfg.samples.close_mm, s))
    dark = fill_holes(dark)

    min_px = int(cfg.samples.min_area_mm2 / s ** 2)
    comps = find_components(dark, min_area_px=min_px)

    masks: list[np.ndarray] = []
    for comp in comps:
        x, y, w, h = comp.bbox
        if x < 0.02 * W and h > 3.5 * w:                       # страховка от линейки
            continue
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(m, [comp.contour], -1, 255, -1)
        masks.append(m)

    warnings: list[str] = []
    exp = cfg.samples.expected
    if exp and masks and len(masks) < exp:
        total = sum(int((m > 0).sum()) for m in masks)
        typical = max(1.0, total / exp)
        split: list[np.ndarray] = []
        for m in masks:
            k = max(1, int(round(float((m > 0).sum()) / typical)))
            if k > 1:
                parts = _split_merged(m, k, cfg, s)
                split.extend(parts)
                if len(parts) != k:
                    warnings.append(f"разделение дало {len(parts)} частей вместо {k}")
            else:
                split.append(m)
        masks = split
        warnings.append(f"применено разделение слипшихся образцов -> {len(masks)}")

    if exp and len(masks) != exp:
        warnings.append(f"найдено {len(masks)} образцов, ожидалось {exp}")
    if not masks:
        warnings.append("образцы не найдены")

    regions: list[SampleRegion] = []
    for m in reading_order(masks, H):
        c, es, el, tilt = sample_basis(m, cfg.frame)
        ys, xs = np.nonzero(m)
        regions.append(SampleRegion(
            mask=m,
            bbox=(int(xs.min()), int(ys.min()),
                  int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
            centroid=(float(c[0]), float(c[1])),
            e_short=es, e_long=el, tilt_deg=float(tilt),
            area_px=int((m > 0).sum()),
        ))
    return regions, warnings
