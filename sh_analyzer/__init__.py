"""Анализатор зон самовосстановления (СВ) металлизированной плёнки.

Публичный API для внешней программы (GUI):
    from sh_analyzer.api import analyze_sync, enqueue_scan, get_result
"""
from .version import __version__, SCHEMA_VERSION

__all__ = ["__version__", "SCHEMA_VERSION"]