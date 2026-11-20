from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from .config import Config


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class Job(BaseModel):
    """Задание на анализ одного скана. Сериализуется в очередь как JSON."""

    job_id: str = Field(default_factory=lambda: uuid4().hex)
    path: str
    out_dir: str | None = None
    config: Config = Field(default_factory=Config)

    sample_labels: dict[str, str] = Field(default_factory=dict)  # "1" -> "PET1400-8a-01"
    session_id: str | None = None
    device: dict = Field(default_factory=dict)          # модель сканера, интерфейс
    scan_profile: dict = Field(default_factory=dict)    # настройки сканирования
    acquired_at: str | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_payload(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: dict) -> Job:
        return cls.model_validate(payload)