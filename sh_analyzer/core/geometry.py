from __future__ import annotations

import math

import cv2
import numpy as np


def mm_to_px(mm: float, mm_per_px: float) -> int:
    """мм -> радиус/размер в пикселях, минимум 1."""
    return max(1, int(round(mm / mm_per_px)))


k_px = mm_to_px  # алиас


def px_to_mm(px: float, mm_per_px: float) -> float:
    return float(px) * mm_per_px


def area_px_to_mm2(area_px: float, mm_per_px: float) -> float:
    return float(area_px) * mm_per_px ** 2


def sample_basis(mask: np.ndarray, mode="sample"):
    """Центроид + базис образца. e_long смотрит 'вверх' (Y изображения вниз).

    mode: строка 'sample'/'scan' либо FrameConfig (берётся .mode).
    """
    mode = getattr(mode, "mode", mode)
    m = (mask > 0).astype(np.uint8)
    mo = cv2.moments(m, binaryImage=True)
    if mo["m00"] == 0:
        raise ValueError("пустая маска образца")
    c = np.array([mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]], dtype=float)

    if mode == "scan":
        e_long = np.array([0.0, -1.0])
    else:
        theta = 0.5 * math.atan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"])
        e_long = np.array([math.cos(theta), math.sin(theta)])
        if e_long[1] > 0:
            e_long = -e_long

    e_short = np.array([-e_long[1], e_long[0]])
    tilt = math.degrees(math.atan2(e_long[0], -e_long[1]))
    return c, e_short, e_long, tilt


def to_frame(pts, c, e_short, e_long, mm_per_px: float) -> np.ndarray:
    """Пиксели скана -> мм в декартовой рамке образца (X вправо, Y вверх)."""
    d = np.atleast_2d(np.asarray(pts, dtype=float)) - np.asarray(c, dtype=float)
    return np.column_stack([d @ e_short * mm_per_px, d @ e_long * mm_per_px])


def reading_order(masks: list[np.ndarray], height: int,
                  row_tol_frac: float = 0.20) -> list[np.ndarray]:
    """Строки сверху вниз, внутри строки слева направо."""
    items = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        items.append((float(ys.mean()), float(xs.mean()), m))
    items.sort(key=lambda t: t[0])

    rows, cur = [], []
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
