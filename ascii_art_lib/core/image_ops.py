"""Общие векторизованные операции над кадрами (утилита, переиспользуется везде).

Раньше одни и те же три строки — «проверить размер и сделать ``cv2.resize``
INTER_AREA», «BGR -> grayscale», «grayscale -> BGR» — дублировались в
``converter``, ``edges.detector``, ``edges.palette``, ``api`` и ``media``.
Здесь они собраны в единственные канонические реализации.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = ["to_gray", "to_bgr", "resize_frame", "needs_resize"]


def to_gray(frame: np.ndarray) -> np.ndarray:
    """BGR ``(H, W, 3)`` uint8 -> grayscale ``(H, W)`` uint8 без лишних копий."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def to_bgr(frame: np.ndarray) -> np.ndarray:
    """Grayscale ``(H, W)`` -> BGR ``(H, W, 3)``; BGR-кадр возвращается как есть."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def needs_resize(frame: np.ndarray, size: Optional[Sequence[int]]) -> bool:
    """True, если кадр ``(h, w)`` не совпадает с целевым размером ``size=(w, h)``."""
    if size is None:
        return False
    return (frame.shape[1], frame.shape[0]) != (int(size[0]), int(size[1]))


def resize_frame(frame: np.ndarray, size: Optional[Sequence[int]]) -> np.ndarray:
    """Приводит кадр к размеру ``(w, h)`` (интерполяция INTER_AREA — быстрый
    качественный даунскейл). ``size=None`` или совпадающий размер — без копии."""
    if not needs_resize(frame, size):
        return frame
    w, h = int(size[0]), int(size[1])  # type: ignore[index]
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def downscale_to_area(frame: np.ndarray, max_pixels: int) -> np.ndarray:
    """Уменьшает кадр, если его площадь больше ``max_pixels`` (защита RAM)."""
    h, w = frame.shape[:2]
    if max_pixels <= 0 or w * h <= max_pixels:
        return frame
    scale = (max_pixels / float(w * h)) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
