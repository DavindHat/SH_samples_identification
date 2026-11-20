from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Инфраструктурные настройки (пути, очередь, логи). Читаются из .env / окружения."""

    model_config = SettingsConfigDict(env_prefix="SH_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    inbox_dir: Path = Path("data/inbox")
    archive_dir: Path = Path("data/archive")
    results_dir: Path = Path("data/results")
    gt_dir: Path = Path("data/gt")

    queue_backend: Literal["sqlite", "redis"] = "sqlite"
    queue_db: Path = Path("data/sh_queue.db")
    redis_url: str = "redis://localhost:6379/0"

    index_db: Path = Path("data/index.db")
    config_path: Path = Path("config/default.yaml")
    default_dpi: float = 300.0
    workers: int = 2

    log_level: str = "INFO"
    log_file: Path = Path("logs/sh_analyzer.log")

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.inbox_dir, self.archive_dir,
                  self.results_dir, self.gt_dir, self.log_file.parent):
            Path(p).mkdir(parents=True, exist_ok=True)


_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = AppSettings()
        _settings.ensure_dirs()
    return _settings