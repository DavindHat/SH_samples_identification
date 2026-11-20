from __future__ import annotations

import cv2
import numpy as np

from ..contracts.config import Config
from ..contracts.result import Sample
from ..core.geometry import mm_to_px


def draw_overlay(bgr: np.ndarray, samples: list[Sample], cfg: Config, mm_per_px: float,
                 active_masks: list[np.ndarray] | None = None) -> np.ndarray:
    out = bgr.copy()
    s = mm_per_px

    for idx, sm in enumerate(samples):
        x, y, w, h = sm.bbox_px
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), max(1, mm_to_px(0.15, s)))

        cx, cy = int(round(sm.centroid_px[0])), int(round(sm.centroid_px[1]))
        cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, max(1, mm_to_px(10, s)), 2)
        cv2.putText(out, f"Sample #{sm.index}", (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Активная зона: полупрозрачный синий слой + оранжевый контур
        if active_masks and idx < len(active_masks):
            act = active_masks[idx]
            if act.any():
                overlay = out.copy()
                overlay[act > 0] = [255, 0, 0]
                cv2.addWeighted(overlay, 0.2, out, 0.8, 0, out)

                cnts, _ = cv2.findContours(act.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, cnts, -1, (255, 100, 0), max(1, mm_to_px(0.3, s)))

        for z in sm.zones:
            color = (0, 0, 255) if "possibly_merged" not in z.flags else (0, 140, 255)

            if z.contour_px and len(z.contour_px) > 2:
                pts = np.array(z.contour_px, dtype=np.int32).reshape(-1, 1, 2)
                cv2.drawContours(out, [pts], -1, color, max(1, mm_to_px(0.15, s)))
            else:
                cv2.circle(out, z.center_px, max(2, mm_to_px(z.d_eq_mm / 2.0, s)), color, max(1, mm_to_px(0.1, s)))

            cv2.circle(out, z.center_px, max(2, mm_to_px(0.3, s)), (0, 255, 255), -1)

            label = f"#{z.id}:{z.area_mm2:.2f}"
            cv2.putText(out, label, (z.center_px[0] + 6, z.center_px[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(out, label, (z.center_px[0] + 6, z.center_px[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    m = cfg.output.overlay_max_side
    if m and max(out.shape[:2]) > m:
        f = m / max(out.shape[:2])
        out = cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

    return out
