from __future__ import annotations

import cv2
import numpy as np

from ..contracts.config import Config
from ..core.geometry import mm_to_px
from ..core.morphology import dilate_disk, erode_disk, open_disk


def active_area(gray: np.ndarray, mask: np.ndarray, cfg: Config, s: float
                ) -> tuple[np.ndarray, float, dict]:
    """Рабочая область образца: внутри кромки, минус контактные полосы и верхняя зона."""
    inner = erode_disk(mask, mm_to_px(cfg.samples.edge_margin_mm, s))
    vals = gray[inner > 0]
    if vals.size == 0:
        return inner > 0, 0.0, {"reason": "пустая внутренняя область"}

    med = float(np.median(vals))
    bands = ((gray < cfg.samples.band_dark_ratio * med) & (inner > 0)).astype(np.uint8) * 255
    bands = open_disk(bands, mm_to_px(cfg.samples.band_open_mm, s))
    bands = dilate_disk(bands, mm_to_px(cfg.samples.band_dilate_mm, s))

    active = (inner > 0) & (bands == 0)

    # Исключаем верхнюю часть образца (где обычно пишут метки)
    if cfg.samples.exclude_top_fraction > 0:
        ys, xs = np.nonzero(mask)
        if ys.size > 0:
            y_min = ys.min()
            y_max = ys.max()
            height = y_max - y_min
            exclude_y = int(y_min + height * cfg.samples.exclude_top_fraction)
            active[:exclude_y, :] = False

    info = {
        "median_gray": round(med, 1),
        "inner_px": int((inner > 0).sum()),
        "band_px": int((bands > 0).sum()),
        "active_px": int(active.sum()),
    }
    return active, med, info
