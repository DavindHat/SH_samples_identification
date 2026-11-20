from pathlib import Path
import sys

EXPECTED = [
    "requirements.txt", "requirements-dev.txt", "pyproject.toml", ".env.example",
    "config/default.yaml",
    "sh_analyzer/__init__.py", "sh_analyzer/version.py", "sh_analyzer/errors.py",
    "sh_analyzer/logging_conf.py", "sh_analyzer/settings.py",
    "sh_analyzer/contracts/__init__.py", "sh_analyzer/contracts/config.py",
    "sh_analyzer/contracts/job.py", "sh_analyzer/contracts/result.py",
    "sh_analyzer/core/__init__.py", "sh_analyzer/core/geometry.py",
    "sh_analyzer/core/imaging.py", "sh_analyzer/core/morphology.py",
    "sh_analyzer/core/components.py",
]

root = Path(__file__).resolve().parent.parent
missing, empty = [], []
for rel in EXPECTED:
    p = root / rel
    if not p.exists():
        missing.append(rel)
    elif p.stat().st_size == 0:
        empty.append(rel)

print(f"Корень: {root}")
print(f"Ожидается {len(EXPECTED)} файлов | нет: {len(missing)} | пустых: {len(empty)}")
for r in missing:
    print("  MISSING:", r)
for r in empty:
    print("  EMPTY  :", r)

for d in ("sh_analyzer", "sh_analyzer/contracts", "sh_analyzer/core", "config"):
    dd = root / d
    if dd.exists():
        odd = [f.name for f in dd.iterdir() if f.is_file() and not f.name.endswith((".py", ".yaml", ".yml"))]
        if odd:
            print(f"  Подозрительные имена в {d}: {odd}")

sys.exit(1 if (missing or empty) else 0)