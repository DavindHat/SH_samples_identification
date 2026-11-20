from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .contracts.config import Config
from .contracts.result import Sample, ScanResult
from .core.geometry import sample_basis
from .core.imaging import file_sha256, load_bgr, resize_to_max
from .detect.active_area import active_area
from .detect.ruler import estimate_scale_from_ruler
from .detect.samples import detect_samples
from .detect.spots import detect_spots
from .measure.framing import spots_to_zones
from .visualize.overlay import draw_overlay


def analyze_scan(path: str | Path, cfg: Config, out_dir: str | Path | None = None) -> ScanResult:
    t0 = time.perf_counter()
    path = Path(path)

    bgr = load_bgr(path)
    orig_h, orig_w = bgr.shape[:2]

    scale_factor = 1.0
    if cfg.perf.resize_max_side and max(orig_h, orig_w) > cfg.perf.resize_max_side:
        bgr, scale_factor = resize_to_max(bgr, cfg.perf.resize_max_side)

    s = cfg.mm_per_px / scale_factor
    t1 = time.perf_counter()

    ruler_warnings: list[str] = []
    if cfg.scale.auto_ruler:
        res, _ = estimate_scale_from_ruler(bgr, cfg, s)
        if res:
            dev = abs(res.mm_per_px / s - 1.0)
            if dev > cfg.scale.ruler_max_deviation:
                ruler_warnings.append(f"Автокалибровка: расхождение {dev:.1%} с заданным DPI")
            s = res.mm_per_px
        else:
            ruler_warnings.append("Автокалибровка по линейке не удалась, используется DPI")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    t2 = time.perf_counter()

    regions, sample_warnings = detect_samples(bgr, cfg, s)
    t3 = time.perf_counter()

    samples_out: list[Sample] = []
    active_masks: list[np.ndarray] = []

    for i, reg in enumerate(regions, 1):
        act_mask, med, act_info = active_area(gray, reg.mask, cfg, s)
        active_masks.append(act_mask)
        c, e_short, e_long, tilt = sample_basis(reg.mask, cfg.frame_mode)

        spots, dbg = detect_spots(bgr, act_mask, cfg, s)
        t4 = time.perf_counter()

        zones = spots_to_zones(
            spots, reg.centroid, e_short, e_long, s,
            cfg.frame.include_contours, cfg.frame.contour_eps_mm,
        )

        area_mm2 = reg.area_px * s * s
        active_mm2 = float(act_mask.sum()) * s * s
        total_zone_mm2 = sum(z.area_mm2 for z in zones)

        warns = list(sample_warnings)
        if any("possibly_merged" in z.flags for z in zones):
            warns.append("Есть зоны, возможно слипшиеся (требуют ручной проверки)")

        samples_out.append(Sample(
            index=i,
            label=None,
            sample_id=None,
            centroid_px=(int(round(reg.centroid[0])), int(round(reg.centroid[1]))),
            tilt_deg=round(tilt, 2),
            area_mm2=round(area_mm2, 2),
            active_area_mm2=round(active_mm2, 2),
            bbox_px=reg.bbox,
            n_zones=len(zones),
            total_zone_area_mm2=round(total_zone_mm2, 4),
            zone_area_fraction=round(total_zone_mm2 / active_mm2, 5) if active_mm2 > 0 else 0.0,
            zones=zones,
            warnings=warns,
        ))

        if cfg.output.save_debug_masks and out_dir:
            d = Path(out_dir) / "debug" / f"{path.stem}_s{i}"
            d.mkdir(parents=True, exist_ok=True)
            for name, m in dbg.items():
                cv2.imwrite(str(d / f"{name}.png"), m)
            cv2.imwrite(str(d / "active.png"), act_mask.astype(np.uint8) * 255)

    t5 = time.perf_counter()

    res = ScanResult(
        file=str(path),
        sha256=file_sha256(path),
        image_size_px=(orig_w, orig_h),
        mm_per_px=s,
        scale_source="ruler" if cfg.scale.auto_ruler else ("manual" if cfg.scale.mm_per_px else "dpi"),
        session_id=None,
        acquired_at=None,
        device={},
        scan_profile={},
        n_samples=len(samples_out),
        n_zones=sum(sm.n_zones for sm in samples_out),
        samples=samples_out,
        warnings=ruler_warnings,
        config=cfg.model_dump(),
        timings_ms={
            "load": round((t1 - t0) * 1e3, 1),
            "ruler": round((t2 - t1) * 1e3, 1),
            "samples": round((t3 - t2) * 1e3, 1),
            "spots": round((t5 - t3) * 1e3, 1),
        },
    )

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        if cfg.output.save_overlay:
            overlay = draw_overlay(bgr, samples_out, cfg, s, active_masks)
            p = out / f"{path.stem}_overlay.png"
            cv2.imwrite(str(p), overlay)
            res.artifacts["overlay"] = str(p)

        if cfg.output.save_json:
            p = out / f"{path.stem}.json"
            res.save_json(p)
            res.artifacts["json"] = str(p)

    return res
