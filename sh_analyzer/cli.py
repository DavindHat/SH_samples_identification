from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .contracts.config import Config
from .core.imaging import list_images
from .pipeline import analyze_scan


def cmd_one(args: argparse.Namespace) -> int:
    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.dpi:
        cfg.scale.dpi = args.dpi
    if args.mm_per_px:
        cfg.scale.mm_per_px = args.mm_per_px
    if args.expected is not None:
        cfg.samples.expected = args.expected
    if args.debug:
        cfg.output.save_debug_masks = True

    try:
        res = analyze_scan(args.input, cfg, args.out)
        print(f"Файл: {res.file}")
        print(f"Масштаб: {res.mm_per_px:.6f} мм/px ({res.scale_source})")
        print(f"Образцов: {res.n_samples}, Зон СВ: {res.n_zones}")
        for sm in res.samples:
            print(f"  Sample #{sm.index}: {sm.n_zones} зон, площадь {sm.total_zone_area_mm2:.3f} мм2")
            if sm.warnings:
                for w in sm.warnings:
                    print(f"    ! {w}")
        if res.artifacts.get("json"):
            print(f"JSON: {res.artifacts['json']}")
        if res.artifacts.get("overlay"):
            print(f"Overlay: {res.artifacts['overlay']}")
        return 0
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1


def cmd_batch(args: argparse.Namespace) -> int:
    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.dpi:
        cfg.scale.dpi = args.dpi
    if args.mm_per_px:
        cfg.scale.mm_per_px = args.mm_per_px
    if args.expected is not None:
        cfg.samples.expected = args.expected
    if args.debug:
        cfg.output.save_debug_masks = True

    in_dir = Path(args.input)
    files = list_images(in_dir)
    if not files:
        print(f"Изображения не найдены в {in_dir}")
        return 1

    print(f"Найдено файлов: {len(files)}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_zones = 0
    for f in files:
        print(f"\n[{files.index(f)+1}/{len(files)}] {f.name}")
        try:
            res = analyze_scan(f, cfg, out_dir)
            print(f"  -> {res.n_samples} образцов, {res.n_zones} зон")
            total_zones += res.n_zones
        except Exception as e:
            print(f"  -> ОШИБКА: {e}")

    print(f"\nИТОГО: обработано {len(files)} файлов, найдено {total_zones} зон СВ.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="sh-analyze", description="Анализатор зон СВ")
    sub = parser.add_subparsers(dest="command", required=True)

    p_one = sub.add_parser("one", help="Анализ одного скана")
    p_one.add_argument("input", help="Путь к изображению")
    p_one.add_argument("--out", default="data/results", help="Папка для результатов")
    p_one.add_argument("--config", help="Путь к YAML конфигу")
    p_one.add_argument("--dpi", type=float, help="DPI сканера")
    p_one.add_argument("--mm-per-px", type=float, dest="mm_per_px", help="Масштаб мм/пиксель")
    p_one.add_argument("--expected", type=int, help="Ожидаемое кол-во образцов")
    p_one.add_argument("--debug", action="store_true", help="Сохранять debug маски")

    p_batch = sub.add_parser("batch", help="Пакетная обработка папки")
    p_batch.add_argument("input", help="Папка со сканами")
    p_batch.add_argument("--out", default="data/results", help="Папка для результатов")
    p_batch.add_argument("--config", help="Путь к YAML конфигу")
    p_batch.add_argument("--dpi", type=float, help="DPI сканера")
    p_batch.add_argument("--mm-per-px", type=float, dest="mm_per_px", help="Масштаб мм/пиксель")
    p_batch.add_argument("--expected", type=int, help="Ожидаемое кол-во образцов")
    p_batch.add_argument("--debug", action="store_true", help="Сохранять debug маски")

    args = parser.parse_args()

    if args.command == "one":
        return cmd_one(args)
    elif args.command == "batch":
        return cmd_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
