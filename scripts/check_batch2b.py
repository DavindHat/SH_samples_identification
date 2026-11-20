"""Синтетическая проверка батча 2B: пятна СВ, halo, автокалибровка по линейке."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sh_analyzer.contracts.config import Config
from sh_analyzer.detect.active_area import active_area
from sh_analyzer.detect.halo import detect_halo
from sh_analyzer.detect.ruler import estimate_scale_from_ruler
from sh_analyzer.detect.samples import detect_samples
from sh_analyzer.detect.spots import detect_spots

FILM = (100, 85, 70)       # R-B = -30
BAND = (15, 15, 15)
WHITE = 235
SPOT = (110, 140, 160)     # R-B = +50
NEUTRAL = (150, 150, 150)  # R-B = 0


def make_scan(s: float):
    H, W = 1600, 1200
    img = np.full((H, W, 3), WHITE, dtype=np.uint8)

    cv2.rectangle(img, (20, 40), (110, 1560), (30, 30, 30), -1)
    y = 60.0
    while y < 1550:
        cv2.line(img, (20, int(y)), (45, int(y)), (230, 230, 230), 1)
        y += 1.0 / s

    cv2.rectangle(img, (350, 250), (850, 800), FILM, -1)
    cv2.rectangle(img, (400, 800), (800, 1350), FILM, -1)
    cv2.rectangle(img, (350, 250), (850, 320), BAND, -1)
    cv2.rectangle(img, (400, 1280), (800, 1350), BAND, -1)

    for yy in (450, 700, 1100):                      # царапины (нейтральные)
        cv2.line(img, (360, yy), (840, yy), (200, 200, 200), 1)
    for xx in range(400, 700, 18):                   # "рукописный текст"
        cv2.line(img, (xx, 345), (xx + 10, 375), (20, 20, 20), 2)

    truth = {}
    cv2.ellipse(img, (600, 500), (25, 18), 0, 0, 360, SPOT, -1)
    truth["big"] = (600, 500)
    cv2.circle(img, (595, 495), 3, (240, 240, 240), -1)   # блики внутри пятна
    cv2.circle(img, (608, 505), 2, (240, 240, 240), -1)

    cv2.circle(img, (500, 1000), 4, SPOT, -1)
    truth["small"] = (500, 1000)

    for p in ((650, 1050), (658, 1053), (652, 1058)):     # кластер = ОДНА зона
        cv2.circle(img, p, 3, SPOT, -1)
    truth["cluster"] = (653, 1054)

    cv2.circle(img, (450, 650), 6, SPOT, -1)
    cv2.circle(img, (700, 650), 6, SPOT, -1)
    truth["left"] = (450, 650)
    truth["right"] = (700, 650)

    cv2.ellipse(img, (700, 900), (30, 20), 0, 0, 360, NEUTRAL, -1)  # только halo
    cv2.circle(img, (200, 900), 5, SPOT, -1)                        # пыль вне образца
    return img, truth


def near(a, b, tol=12.0):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def main() -> int:
    cfg = Config.from_yaml("config/default.yaml")
    s = cfg.mm_per_px
    out = Path("data/debug"); out.mkdir(parents=True, exist_ok=True)
    img, truth = make_scan(s)
    ok = True

    regions, warns = detect_samples(img, cfg, s)
    print(f"образцов: {len(regions)}")
    if len(regions) != 1:
        print("FAIL: ожидался 1 образец"); return 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    act, med, info = active_area(gray, regions[0].mask, cfg, s)
    spots, dbg = detect_spots(img, act, cfg, s)

    print(f"\nзон СВ найдено: {len(spots)} (ожидалось 5)")
    ok &= len(spots) == 5
    total = 0.0
    for i, z in enumerate(spots, 1):
        a = z.area_px * s * s
        total += a
        print(f"  #{i}: ({z.center_px[0]:6.1f},{z.center_px[1]:7.1f}) "
              f"площадь {a:7.3f} мм2 | warm {z.warm_mean:5.1f} "
              f"| circ {z.circularity:.3f} | solid {z.solidity:.3f} "
              f"| conf {z.confidence:.2f} | {z.flags}")
    print(f"  суммарная площадь: {total:.3f} мм2")
    ok &= 8.0 <= total <= 20.0

    for name, p in truth.items():
        hit = [z for z in spots if near(z.center_px, p)]
        good = len(hit) == 1
        print(("  OK: " if good else "  FAIL: ") + f"зона '{name}' найдена ровно один раз")
        ok &= good

    big = [z for z in spots if near(z.center_px, truth["big"])]
    if big:
        a = big[0].area_px * s * s
        good = 8.5 <= a <= 11.8
        print(("  OK: " if good else "  FAIL: ") +
              f"площадь большого пятна {a:.3f} мм2 (эталон ~10.1, блики заполнены)")
        ok &= good

    dust = [z for z in spots if near(z.center_px, (200, 900), tol=20)]
    print(("  OK: " if not dust else "  FAIL: ") + "пыль вне образца не посчитана")
    ok &= not dust

    halos, hdbg = detect_halo(img, act, cfg, s)
    neutral_hit = [h for h in halos if near(h.centroid, (700, 900), tol=20)]
    print(f"\nhalo-компонент: {len(halos)}")
    print(("  OK: " if neutral_hit else "  FAIL: ") +
          "нейтральное светлое поле видно в halo (и не видно в spots)")
    ok &= bool(neutral_hit)

    res, rinfo = estimate_scale_from_ruler(img, cfg, s)
    print("\nлинейка:")
    if res is None:
        print("  FAIL:", rinfo.get("reason")); ok = False
    else:
        dev = abs(res.mm_per_px / s - 1.0)
        print(f"  мм/px {res.mm_per_px:.6f} | рисок {res.n_ticks} | шаг {res.spacing_px:.3f} px "
              f"| единица {res.unit_mm} мм | наклон {res.angle_deg:.1f} | CV {res.spacing_cv:.3f} "
              f"| откл от DPI {dev:.2%} | conf {res.confidence:.2f} | кромка {res.band}")
        good = dev < 0.01 and res.n_ticks > 100 and res.unit_mm == 1.0
        print(("  OK: " if good else "  FAIL: ") + "масштаб совпал с эталонным")
        ok &= good
        vis = cv2.cvtColor(rinfo["strip_rot"], cv2.COLOR_GRAY2BGR)
        xa, xb = rinfo["band_x"]
        for py in rinfo["peaks"]:
            cv2.line(vis, (xa, int(py)), (xb, int(py)), (0, 0, 255), 1)
        cv2.imwrite(str(out / "synthetic_2b_ruler.png"), vis)

    vis = img.copy()
    cv2.drawContours(vis, [regions[0].contour if hasattr(regions[0], "contour") else
                           cv2.findContours(regions[0].mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)[0][0]],
                     -1, (0, 200, 0), 2)
    for i, z in enumerate(spots, 1):
        cv2.drawContours(vis, [z.contour], -1, (0, 0, 255), 2)
        cx, cy = int(z.center_px[0]), int(z.center_px[1])
        cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)
        cv2.putText(vis, f"{i}:{z.area_px * s * s:.2f}", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(out / "synthetic_2b_overlay.png"), vis)
    for k, v in dbg.items():
        cv2.imwrite(str(out / f"synthetic_2b_{k}.png"), v)
    cv2.imwrite(str(out / "synthetic_2b_halo.png"), hdbg["halo"])

    print("\nИТОГ:", "OK" if ok else "ЕСТЬ ОШИБКИ")
    print("картинки:", out.resolve())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
