from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from ..errors import ImageLoadError

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageLoadError(f"Не читается изображение: {path}")
    return img


def save_image(path: str | Path, img: np.ndarray) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(p), img):
        raise ImageLoadError(f"Не удалось сохранить: {p}")
    return str(p)


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def gray_of(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def warmth(bgr: np.ndarray) -> np.ndarray:
    """Канал R-B: металлизация отрицательная, пятно прожога положительное."""
    b, _, r = cv2.split(bgr.astype(np.int16))
    return r - b


def resize_to_max(img: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float]:
    """Возвращает (изображение, factor). factor<1 означает уменьшение."""
    if not max_side or max(img.shape[:2]) <= max_side:
        return img, 1.0
    f = max_side / max(img.shape[:2])
    out = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return out, f


def to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def flat_field_correct(bgr: np.ndarray, flat_bgr: np.ndarray) -> np.ndarray:
    """Коррекция неравномерности освещения по скану пустой белой подложки."""
    if flat_bgr.shape != bgr.shape:
        flat_bgr = cv2.resize(flat_bgr, (bgr.shape[1], bgr.shape[0]),
                              interpolation=cv2.INTER_AREA)
    flat = cv2.GaussianBlur(flat_bgr.astype(np.float32), (0, 0), 15)
    mean = flat.reshape(-1, 3).mean(axis=0)
    gain = mean / np.maximum(flat, 1.0)
    out = bgr.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXT)