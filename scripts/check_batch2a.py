"""Синтетическая проверка батча 2A: поиск образцов, рабочая область, разделение."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sh_analyzer.contracts.config import Config
from sh_analyzer.detect.active_area import active_area
from sh_analyzer.detect.samples import detect_samples

FILM = (100, 85, 70)      # BGR: сине-серая металлизация, R-B = -30
BAND = (15, 15, 15)
WHITE = 235


def _sample(img, x0, y0, x1, y1, waist=50):
    mid = (y0 + y1) // 2
    cv2.rectangle(img, (x0, y0), (x1, mid), FILM, -1)
    cv2.rectangle(img, (x0 + waist, mid), (x1 - waist, y1), FILM, -1)
    cv2.rectangle(img, (x0, y0), (x1, y0 + 70), BAND, -1)
    cv2.rectangle(img, (x0 + waist, y1 - 70), (x1 - waist, y1), BAND, -1)


def make_scan(kind: str, s: float) -> np.ndarray:
    H, W = 1600, 1200
    img = np.full((H, W, 3), WHITE, dtype=np.uint8)

    cv2.rectangle(img, (20, 40), (110, 1560), (30, 30, 30), -1)
    y = 60.0
    while y < 1550:
        cv2.line(img, (20, int(y)), (45, int(y)), (230, 230, 230), 1)
        y += 1.0 / s

    if kind == "single":
        _sample(img, 350, 250, 850, 1350)
    elif kind == "quad":
        _sample(img, 300, 150, 560, 780)
        _sample(img, 640, 150, 900, 780)
        _sample(img, 300, 850, 560, 1480)
        _sample(img, 640, 850, 900, 1480)
    elif kind == "bridged":
        _sample(img, 300, 200, 560, 1400, waist=0)
        _sample(img, 640, 200, 900, 1400, waist=0)
        cv2.rectangle(img, (555, 700), (645, 720), FILM, -1)   # мостик 20 px
    return img


def draw(vis, regions, s):
    for i, r in enumerate(regions, 1):
        x, y, w, h = r.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 3)
        cnts, _ = cv2.findContours(r.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 0, 0), 2)
        cx, cy = int(round(r.centroid[0])), int(round(r.centroid[1]))
        cv2.drawMarker(vis, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
        L = int(round(10.0 / s))
        for vec, col in ((r.e_short, (255, 128, 0)), (r.e_long, (255, 0, 255))):
            cv2.arrowedLine(vis, (cx, cy),
                            (int(cx + vec[0] * L), int(cy + vec[1] * L)),
                            col, 3, tipLength=0.2)
        cv2.putText(vis, f"#{i}", (x + 10, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)


def main() -> int:
    cfg = Config.from_yaml("config/default.yaml")
    s = cfg.mm_per_px
    print(f"масштаб: {s:.6f} мм/px")
    out = Path("data/debug")
    out.mkdir(parents=True, exist_ok=True)
    ok = True

    cases = [("single", 1, 3000, 4200), ("quad", 4, 700, 1500), ("bridged", 2, 1500, 3000)]

    for kind, expect, amin, amax in cases:
        c = Config.from_yaml("config/default.yaml")
        if kind == "bridged":
            c.samples.expected = 2
        img = make_scan(kind, s)
        regions, warns = detect_samples(img, c, s)

        print(f"\n--- {kind}: задано {expect}, найдено {len(regions)} ---")
        for w in warns:
            print("  warning:", w)
        if len(regions) != expect:
            print("  FAIL: количество не совпало")
            ok = False

        for i, r in enumerate(regions, 1):
            a = r.area_px * s * s
            print(f"  #{i}: площадь {a:8.1f} мм2 | наклон {r.tilt_deg:6.2f} deg "
                  f"| центроид ({r.centroid[0]:.0f}, {r.centroid[1]:.0f}) | bbox {r.bbox}")
            if abs(r.tilt_deg) > 3.0:
                print("  FAIL: неожиданный наклон")
                ok = False
            if not (amin <= a <= amax):
                print(f"  FAIL: площадь вне диапазона {amin}..{amax}")
                ok = False

        if kind == "quad" and len(regions) == 4:
            cx = [r.centroid[0] for r in regions]
            cy = [r.centroid[1] for r in regions]
            if cx[0] < cx[1] and cx[2] < cx[3] and cy[0] < cy[2] and cy[1] < cy[3]:
                print("  OK: порядок чтения (ЛВ, ПВ, ЛН, ПН)")
            else:
                print("  FAIL: порядок чтения нарушен")
                ok = False

        if kind == "single" and regions:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            act, med, info = active_area(gray, regions[0].mask, c, s)
            print("  рабочая область:", info)
            for label, cond in (
                ("середина плёнки активна", bool(act[700, 600])),
                ("контактная полоса исключена", not bool(act[270, 600])),
                ("кромка исключена", not bool(act[255, 360])),
            ):
                print(("  OK: " if cond else "  FAIL: ") + label)
                ok = ok and cond
            cv2.imwrite(str(out / "synthetic_2a_active.png"), act.astype(np.uint8) * 255)

        vis = img.copy()
        draw(vis, regions, s)
        cv2.imwrite(str(out / f"synthetic_2a_{kind}.png"), vis)

    print("\nИТОГ:", "OK" if ok else "ЕСТЬ ОШИБКИ")
    print("картинки:", out.resolve())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
