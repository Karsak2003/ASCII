"""Оптимизированное по памяти итеративное чтение кадров изображений/видео/GIF.

Главная проблема исходного кода: все кадры видео/GIF загружались в список
``list[np.ndarray]`` целиком, что для больших файлов съедале гигабайты RAM.
Здесь вместо этого предоставляются генераторы, которые читают **по одному кадру**
и немедленно отдают его потребителю (конвертеру), после чего кадр может быть
собран сборщиком мусора.
"""

from __future__ import annotations

import os
from typing import Iterator, NamedTuple, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageSequence

from ascii_art_lib.core.image_ops import downscale_to_area


class MediaInfo(NamedTuple):
    """Метаданные медиа-файла (без хранения самих кадров)."""

    path: str
    kind: str            # "image" | "animation"
    width: int
    height: int
    fps: float           # 0.0 для статичных изображений
    n_frames: int        # 1 для статичных изображений


_EXT_IMAGE = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
_EXT_ANIM = {"gif", "mp4", "avi", "mov", "mkv", "webm", "m4v"}


def classify(path: str) -> str:
    """Возвращает ``"image"`` или ``"animation"`` по расширению файла."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext == "gif":
        return "animation"
    if ext in _EXT_IMAGE:
        return "image"
    if ext in _EXT_ANIM or ext in {"mp4", "avi", "mov", "mkv", "webm"}:
        return "animation"
    # Неизвестное расширение — пробуем как видео через OpenCV
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened()
    cap.release()
    return "animation" if ok else "image"


def probe(path: str) -> MediaInfo:
    """Считывает только метаданные файла (не грузя пиксели в память)."""
    kind = classify(path)
    if kind == "animation":
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть медиа-файл: {path!r}")
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if n <= 0:  # некоторые контейнеры не сообщают длину заранее
            n = 1
        return MediaInfo(path=path, kind="animation", width=w, height=h, fps=fps, n_frames=n)

    # Статичное изображение (GIF обрабатывается выше как анимация)
    with Image.open(path) as im:
        w, h = im.size
    return MediaInfo(path=path, kind="image", width=w, height=h, fps=0.0, n_frames=1)


def iter_frames(path: str, *, max_pixels: Optional[int] = None) -> Iterator[np.ndarray]:
    """Лениво отдаёт кадры изображения/анимации в формате BGR ``uint8``.

    Память: в каждый момент времени живёт не более одного полного кадра.

    Args:
        path: Путь к файлу (изображение, GIF или видео).
        max_pixels: Если задан, кадры с бо́льшим числом пикселей предварительно
            уменьшаются (``cv2.resize`` INTER_AREA) до этой площади. Это резко
            снижает пиковое потребление RAM для огромных источников.
    """
    kind = classify(path)

    if kind == "animation" and path.lower().endswith(".gif"):
        yield from _iter_gif_frames(path, max_pixels=max_pixels)
    elif kind == "animation":
        yield from _iter_video_frames(path, max_pixels=max_pixels)
    else:
        frame = _load_image_bgr(path)
        if max_pixels and frame.shape[0] * frame.shape[1] > max_pixels:
            frame = downscale_to_area(frame, max_pixels)
        yield frame


def _load_image_bgr(path: str) -> np.ndarray:
    """Читает одиночное изображение сразу в BGR uint8 (без лишних копий)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Не удалось прочитать изображение: {path!r}")
    return img


def _iter_image_bgr(path: str) -> Iterator[np.ndarray]:
    """Гарантированно освобождает буфер PIL между кадрами."""
    im = Image.open(path)
    try:
        for frame in ImageSequence.Iterator(im):
            rgb = frame.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
            # RGB->BGR без лишней копии через обратный шаг по оси каналов
            yield np.ascontiguousarray(arr[:, :, ::-1])
    finally:
        im.close()


def _iter_gif_frames(path: str, *, max_pixels: Optional[int] = None) -> Iterator[np.ndarray]:
    for frame in _iter_image_bgr(path):
        if max_pixels and frame.shape[0] * frame.shape[1] > max_pixels:
            frame = downscale_to_area(frame, max_pixels)
        yield frame


def _iter_video_frames(path: str, *, max_pixels: Optional[int] = None) -> Iterator[np.ndarray]:
    """Читает видео покадрово через cv2.VideoCapture (одна страница кадра в RAM)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {path!r}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if max_pixels and frame.shape[0] * frame.shape[1] > max_pixels:
                frame = downscale_to_area(frame, max_pixels)
            yield frame
    finally:
        cap.release()


def get_fps(path: str) -> float:
    """Возвращает FPS анимации (или 0 для статичных изображений)."""
    return probe(path).fps
