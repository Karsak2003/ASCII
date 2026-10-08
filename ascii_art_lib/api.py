"""Публичный API пакета: высокоуровневые функции конвертации и вывода.

Здесь живут основные точки входа, которые использует как CLI, так и сторонний код:

* :func:`convert_image`  — изображение -> ASCII-строка (или список строк-строк).
* :func:`convert_animation` — GIF/видео -> **ленивый генератор** ASCII-кадров
  (память O(1 кадр), а не O(N кадров)).
* :func:`show_image` / :func:`play_animation` — конвертация + вывод в консоль
  через :class:`~ascii_art_lib.renderer.ConsoleRenderer`.
* :func:`save_ascii` — сохранение результата в файл (.txt / .ans).

Все функции используют внутри себя методы из :mod:`ascii_art_lib.converter`
и :mod:`ascii_art_lib.media`, поэтому интерфейс вызова остаётся простым.
"""

from __future__ import annotations

import os
import shutil
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .converter import frame_to_color_ansi, frame_to_mono_text
from .media import classify, iter_frames, probe
from .palettes import DEFAULT_PALETTE, get_palette
from .renderer import ConsoleRenderer

__all__ = [
    "convert_image",
    "convert_animation",
    "show_image",
    "play_animation",
    "save_ascii",
    "terminal_fit_size",
]


# ---------------------------------------------------------------------------
# Вспомогательное: подбор размера под терминал с сохранением пропорций
# ---------------------------------------------------------------------------

def terminal_fit_size(
    src_w: int,
    src_h: int,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    char_aspect: float = 2.0,
) -> Tuple[int, int]:
    """Возвращает ``(w, h)`` в символах для консоли с сохранением пропорций.

    Args:
        src_w/src_h: Размеры источника в пикселях.
        width/height: Ограничения области вывода (по умолчанию — размер терминала).
        char_aspect: Отношение высоты символа к его ширине (обычно ~2.0).
    """
    term_w, term_h = shutil.get_terminal_size((80, 24))
    max_w = int(width or term_w)
    max_h = int(height or (term_h - 1))
    if src_w <= 0 or src_h <= 0:
        return max_w, max_h
    aspect = src_h / src_w
    w = min(max_w, int(char_aspect * max_h / aspect))
    h = max(1, min(max_h, int(w * aspect / char_aspect)))
    return max(2, w), h


def _resolve_size(
    size: Optional[Sequence[int]],
    src_w: int,
    src_h: int,
    fullcolor: bool,
) -> Tuple[int, int]:
    if size is not None:
        w, h = int(size[0]), int(size[1])
        if w > 0 and h > 0:
            return w, h
    # Автоподбор под терминал; для цветного режима чуть уменьшаем высоту,
    # чтобы оставить место под ANSI-префиксы и строку статуса.
    return terminal_fit_size(src_w, src_h, height=None if not fullcolor else None)


# ---------------------------------------------------------------------------
# Конвертация статичного изображения
# ---------------------------------------------------------------------------

def convert_image(
    source: Union[str, np.ndarray],
    *,
    palette: str = DEFAULT_PALETTE,
    size: Optional[Tuple[int, int]] = None,
    fullcolor: bool = True,
    color_levels: int = 16,
    invert: bool = False,
    max_pixels: Optional[int] = 32_000_000,
) -> str:
    """Конвертирует изображение (путь или ``np.ndarray`` BGR) в ASCII-строку.

    Args:
        source: Путь к файлу либо уже загруженный кадр ``ndarray`` (H, W, 3) uint8.
        palette: Имя палитры (``"asii"``, ``"asii_3v"`` …) или пользовательская строка
            символов от тёмных к светлым.
        size: Целевой размер ``(ширина, высота)`` в символах. ``None`` — автоподбор
            под терминал с сохранением пропорций.
        fullcolor: ``True`` — цветной ANSI truecolor, ``False`` — монохром.
        color_levels: Уровней квантования на канал при ``fullcolor=True``.
        invert: Инвертировать яркость (светлое/тёмное).
        max_pixels: Предварительно уменьшить источник, если он больше этой площади
            (защита RAM для гигантских файлов). ``None`` — без ограничения.

    Returns:
        Готовая многострочная строка ASCII-арта.
    """
    frame = _load_frame(source, max_pixels=max_pixels)
    if invert:
        frame = 255 - frame

    info_w, info_h = frame.shape[1], frame.shape[0]
    w, h = _resolve_size(size, info_w, info_h, fullcolor)

    if fullcolor:
        return frame_to_color_ansi(frame, palette, (w, h), color_levels=color_levels)
    return frame_to_mono_text(frame, palette, (w, h))


# ---------------------------------------------------------------------------
# Конвертация анимации (GIF / видео) — ленивый генератор, экономия RAM
# ---------------------------------------------------------------------------

def convert_animation(
    path: str,
    *,
    palette: str = DEFAULT_PALETTE,
    size: Optional[Tuple[int, int]] = None,
    fullcolor: bool = True,
    color_levels: int = 16,
    invert: bool = False,
    max_pixels: Optional[int] = 32_000_000,
    progress: bool = False,
) -> Iterator[str]:
    """Конвертирует GIF/видео в **генератор** ASCII-кадров.

    Кадры читаются и конвертируются по одному (streaming), поэтому потребление
    памяти не зависит от длительности анимации — это ключевая оптимизация RAM
    по сравнению с исходным «загрузить все кадры в список».

    Args: см. :func:`convert_image`. Дополнительно:
        progress: Показывать инкрементный прогресс-бар во время конвертации
            (если доступна библиотека ``progress``).

    Yields:
        str: очередной ASCII-кадр (ANSI, если ``fullcolor=True``).
    """
    info = probe(path)
    w, h = _resolve_size(size, info.width, info.height, fullcolor)

    frames_iter = iter_frames(path, max_pixels=max_pixels)

    bar = None
    if progress:
        try:
            from progress.bar import IncrementalBar

            bar = IncrementalBar("Converting", max=max(info.n_frames, 1))
        except Exception:
            bar = None

    for frame in frames_iter:
        if invert:
            frame = 255 - frame
        if fullcolor:
            text = frame_to_color_ansi(frame, palette, (w, h), color_levels=color_levels)
        else:
            text = frame_to_mono_text(frame, palette, (w, h))
        if bar is not None:
            bar.next()
        yield text
        del frame  # немедленно отпускаем кадр — следующий ещё не прочитан

    if bar is not None:
        bar.finish()


# ---------------------------------------------------------------------------
# Вывод в консоль
# ---------------------------------------------------------------------------

def show_image(
    source: Union[str, np.ndarray],
    *,
    renderer: Optional[ConsoleRenderer] = None,
    header: str = "",
    save_path: Optional[str] = None,
    **convert_kwargs,
) -> str:
    """Конвертирует изображение и сразу печатает его в консоль.

    Возвращает сгенерированную строку (удобно для тестов/программного использования).
    """
    text = convert_image(source, **convert_kwargs)
    r = renderer or ConsoleRenderer()
    r.show_static(text, header=header)
    if save_path:
        save_ascii(text, save_path)
    return text


def play_animation(
    path: str,
    *,
    renderer: Optional[ConsoleRenderer] = None,
    fps: Optional[float] = None,
    duration: Optional[float] = None,
    status_header: bool = True,
    save_dir: Optional[str] = None,
    **convert_kwargs,
) -> None:
    """Проигрывает GIF/видео как ASCII-анимацию в консоли (стриминг кадров)."""
    info = probe(path)
    r = renderer or ConsoleRenderer()
    eff_fps = float(fps or info.fps or 25.0)

    gen = convert_animation(path, **convert_kwargs)

    def header_fn(idx: int, loop: int) -> str:
        if not status_header:
            return ""
        return f"{os.path.basename(path)} | {info.width}x{info.height} | FPS:{eff_fps:.1f} | frame {idx}"

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def tee(it: Iterator[str]) -> Iterator[str]:
        for i, t in enumerate(it):
            if save_dir:
                r.save(t, os.path.join(save_dir, f"frame_{i:05d}.ans"))
            yield t

    r.play(tee(gen), fps=eff_fps, duration=duration, header_fn=header_fn)


# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------

def save_ascii(text: str, path: str) -> str:
    """Сохраняет ASCII/ANSI-строку в файл; возвращает путь."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_frame(source: Union[str, np.ndarray], *, max_pixels: Optional[int]) -> np.ndarray:
    if isinstance(source, np.ndarray):
        frame = source
        if frame.ndim == 2:
            import cv2

            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame
    if isinstance(source, (str, os.PathLike)):
        info = classify(str(source))
        it = iter_frames(str(source), max_pixels=max_pixels)
        frame = next(it, None)
        if frame is None:
            raise IOError(f"Не удалось прочитать изображение: {source!r}")
        return frame
    raise TypeError(f"source должен быть str/os.PathLike/np.ndarray, получено {type(source)!r}")
