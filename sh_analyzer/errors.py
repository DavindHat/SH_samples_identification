from __future__ import annotations


class SHError(Exception):
    """Базовая ошибка анализатора."""


class ConfigError(SHError):
    """Некорректный конфиг или профиль."""


class ScaleError(SHError):
    """Не удалось определить масштаб (мм/пиксель)."""


class ImageLoadError(SHError):
    """Файл не читается как изображение."""


class SampleDetectionError(SHError):
    """Образцы не найдены или найдены некорректно."""


class JobError(SHError):
    """Ошибка выполнения задачи."""