"""Выделение контуров (edge detection) для ASCII-арта.

Реализация опирается на классический pipeline из видео
(https://youtu.be/gg40RWiaHRY — «Canny Edge Detection» / sobel-based edge
detectors):

    1. Gaussain blur   — подавление шума (cv2.GaussianBlur);
    2. Sobel Gx, Gy    — градиенты яркости по горизонтали/вертикали
       (используются готовые кэшированные ядра :mod:`ascii_art_lib.threshold_map`);
    3. Magnitude       — ``sqrt(Gx^2 + Gy^2)`` (быстрая аппроксимация через
       ``cv2.magnitude`` — C-реализация, без Python-циклов);
    4. Normalization   — приведение ``0..255``;
    5. Thresholding    — двойной порог с гистерезисом: сильные пиксели —
       гарантированные контуры, слабые — только если соединяются с сильными
       (аналог ``cv2.Canny`` low/high threshold, но на собственном magnitude).

Все операции векторизованы (OpenCV/NumPy), поэтому стоимость — O(H*W) на
стороне C: это и оптимизация производительности, и экономия RAM (промежуточные
массивы ``float32``/``uint8`` освобождаются внутри функции).

Модуль предоставляет:

* :func:`detect_edges` — низкоуровневая функция (кадр -> uint8-карта контуров);
* :class:`EdgeDetector` — конфигурируемый детектор (метод ``canny``/``sobel``,
  пороги, размытие), удобно переиспользовать в цикле по кадрам анимации;
* :func:`blend_with_source` — смешивание карты контуров с оригиналом (режимы
  ``"lines"`` — только линии, ``"overlay"`` — контур поверх изображения).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ascii_art_lib.core.image_ops import resize_frame, to_gray
from ascii_art_lib.core.threshold_map import Gx as SOBEL_GX
from ascii_art_lib.core.threshold_map import Gy as SOBEL_GY

__all__ = ["EdgeDetector", "detect_edges", "blend_with_source"]


# ---------------------------------------------------------------------------
# Ядро: карта контуров
# ---------------------------------------------------------------------------

def detect_edges(
    frame: np.ndarray,
    *,
    method: str = "canny",
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_ksize: int = 5,
    size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Возвращает uint8-карту контуров ``(H, W)`` со значениями 0 или 255.

    Args:
        frame: BGR (H, W, 3) или grayscale (H, W) uint8.
        method: ``"canny"`` (полный pipeline: blur -> sobel -> magnitude ->
            double-threshold с гистерезисом) либо ``"sobel"`` (упрощённый
            вариант без гистерезиса — один порог ``low_threshold``).
        low_threshold: Нижний порог величины градиента (0..255).
        high_threshold: Верхний порог (гарантированные контуры). Игнорируется
            в режиме ``"sobel"``.
        blur_ksize: Размер гауссова ядра предварительного размытия (0 — выключить).
        size: Если задан ``(w, h)``, контуры считаются уже на уменьшенном кадре
            (быстрее и меньше RAM для больших изображений).

    Returns:
        np.ndarray (H, W) dtype uint8, 0 — фон, 255 — контур.
    """
    frame = resize_frame(frame, size)

    gray = to_gray(frame)

    # 1. Подавление шума
    if blur_ksize and blur_ksize >= 3:
        k = int(blur_ksize) | 1  # нечётное
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # 2-3. Градиенты и величина (C-скорость, float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, SOBEL_GX)
    gy = cv2.filter2D(gray, cv2.CV_32F, SOBEL_GY)
    mag = np.empty_like(gx)
    cv2.magnitude(gx, gy, mag)
    del gx, gy

    # 4. Нормализация 0..255
    mmax = float(mag.max()) if mag.size else 0.0
    if mmax > 255.0:
        mag *= 255.0 / mmax
    mag_u8 = mag.astype(np.uint8)
    del mag

    # 5. Пороговая обработка
    if method == "sobel":
        _, edges = cv2.threshold(mag_u8, int(low_threshold), 255, cv2.THRESH_BINARY)
        return edges

    # Гистерезис (double threshold) — аналог финального шага Canny:
    # сильные пиксели оставляем, слабые — только если они связаны с сильными.
    strong = mag_u8 >= high_threshold
    weak = mag_u8 >= low_threshold
    if not strong.any():
        # Нет гарантированных контуров — откатываемся к простому порогу,
        # иначе результат был бы пустым.
        _, edges = cv2.threshold(mag_u8, int(low_threshold), 255, cv2.THRESH_BINARY)
        return edges
    if not weak.any():
        return strong.astype(np.uint8) * 255

    # Итеративное распространение «силы» по слабым пикселям (морфологическая
    # реконструкция). Число проходов ограничено размером изображения по диагонали,
    # на практике достаточно нескольких десятков; используем побитовые операции
    # над массивами целиком (векторно).
    seed = strong.astype(np.uint8) * 255  # cv2.dilate не поддерживает bool (dtype=9)
    kernel = np.ones((3, 3), np.uint8)
    # Ограничиваем число итераций, чтобы на огромных файлах не зависнуть:
    max_iters = min(64, max(seed.shape) // 2 + 1)
    for _ in range(max_iters):
        dilated = cv2.dilate(seed, kernel, iterations=1)
        candidate = np.where(weak & (dilated > 0), 255, 0).astype(np.uint8)
        if np.array_equal(candidate, seed):
            break
        seed = candidate
    edges = seed
    return edges


# ---------------------------------------------------------------------------
# Конфигурируемый объект (для переиспользования настроек в цикле кадров)
# ---------------------------------------------------------------------------

class EdgeDetector:
    """Класс-конфигуратор выделения контуров.

    Пример::

        det = EdgeDetector(method="canny", low_threshold=40, high_threshold=120)
        for frame in iter_frames("clip.mp4"):
            edges = det.detect(frame)
    """

    def __init__(
        self,
        *,
        method: str = "canny",
        low_threshold: int = 50,
        high_threshold: int = 150,
        blur_ksize: int = 5,
    ) -> None:
        if method not in ("canny", "sobel"):
            raise ValueError(f"Неизвестный метод детекции контуров: {method!r}")
        if low_threshold > high_threshold:
            low_threshold, high_threshold = high_threshold, low_threshold
        self.method = method
        self.low_threshold = int(low_threshold)
        self.high_threshold = int(high_threshold)
        self.blur_ksize = int(blur_ksize)

    def detect(self, frame: np.ndarray, *, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Карта контуров для одного кадра (см. :func:`detect_edges`)."""
        return detect_edges(
            frame,
            method=self.method,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
            blur_ksize=self.blur_ksize,
            size=size,
        )

    def apply(
        self,
        frame: np.ndarray,
        *,
        mode: str = "lines",
        color: Tuple[int, int, int] = (255, 255, 255),
        alpha: float = 0.5,
        size: Optional[Tuple[int, int]] = None,
        curve_palette_mode: str = "extended",
        curve_threshold: float = 0.5,
    ) -> np.ndarray:
        """Возвращает кадр, готовый к ASCII-конвертации с контурами.

        Args:
            mode: ``"lines"`` — чёрно-белое изображение, где контуры белые на
                чёрном фоне (чистый edge-art); ``"overlay"`` — контуры, наложенные
                на оригинал (оригинал затемняется в ``alpha`` раз в местах линий);
                ``"curves"``/``"lines"``/``"palette"`` — спец-режим «палитры
                ориентации»: возвращаются
                **ASCII-байты символов наклона** (uint8 HxW, см.
                :func:`~ascii_art_lib.edge_palette.frame_to_edge_symbols`) —
                такой выход подаётся напрямую в ``edge_symbols_to_text``,
                а не в обычные яркостные конвертеры.
            color: Цвет линий для ``overlay`` (BGR).
            alpha: Сила затемнения оригинала под линиями (0..1).
            curve_palette_mode / curve_threshold: параметры палитры ориентации
                для режимов ``"curves"``/``"lines"``/``"palette"``
                (``"curves"`` → ``extended``, ``"lines"`` → ``basic``).
        """
        frame = resize_frame(frame, size)
        if mode in ("palette", "curves", "lines"):
            from .palette import frame_to_edge_symbols

            if mode == "curves":
                curve_palette_mode = "extended"
            elif mode == "lines":
                curve_palette_mode = "basic"
            return frame_to_edge_symbols(
                frame, None,
                mode=curve_palette_mode,
                low_threshold=self.low_threshold,
                high_threshold=self.high_threshold,
                blur_ksize=self.blur_ksize,
                curve_threshold=curve_threshold,
                method=self.method,
            )
        edges = self.detect(frame)
        mask = edges > 0
        if mode == "lines":
            out = np.zeros_like(frame)
            out[mask] = (255, 255, 255)
            return out
        if mode == "overlay":
            out = frame.copy()
            out[mask] = color
            return out
        raise ValueError(
            f"Неизвестный режим наложения: {mode!r} "
            "('lines'/'overlay'/'curves'/'palette')"
        )


# ---------------------------------------------------------------------------
# Смешивание с оригиналом (общая утилита)
# ---------------------------------------------------------------------------

def blend_with_source(
    frame: np.ndarray,
    edges: np.ndarray,
    *,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Накладывает готовую карту контуров поверх кадра (возвращает новый кадр)."""
    out = frame.copy()
    mask = edges > 0
    out[mask] = color
    return out
