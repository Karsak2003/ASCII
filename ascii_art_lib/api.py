"""Публичный API пакета: высокоуровневые функции конвертации и вывода.

Здесь живут основные точки входа, которые использует как CLI, так и сторонний код:

* :func:`convert_image`  — изображение -> ASCII-строка (или список строк-строк).
* :func:`convert_animation` — GIF/видео -> **ленивый генератор** ASCII-кадров
  (память O(1 кадр), а не O(N кадров)).
* :func:`show_image` / :func:`play_animation` — конвертация + вывод в консоль
  через :class:`~ascii_art_lib.rendering.console.ConsoleRenderer`.
* :func:`save_ascii` — сохранение результата в файл (.txt / .ans).

Все функции используют внутри себя методы из :mod:`ascii_art_lib.core.converter`
и :mod:`ascii_art_lib.core.media`, поэтому интерфейс вызова остаётся простым.
"""

from __future__ import annotations

import os
import shutil
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .core.converter import frame_to_color_ansi, frame_to_mono_text
from .core.media import classify, iter_frames, probe
from .core.palettes import DEFAULT_PALETTE
from .edges import EdgeDetector
from .edges.palette import normalize_fill
from .edges.pipeline import convert_edge_frame
from .rendering import ConsoleRenderer

__all__ = [
    "convert_image",
    "convert_animation",
    "show_image",
    "play_animation",
    "save_ascii",
    "terminal_fit_size",
]


def _make_detector(
    edges: Union[bool, str, None],
    *,
    edge_mode: str,
    low_threshold: int,
    high_threshold: int,
    blur_ksize: int,
) -> Optional[EdgeDetector]:
    """Собирает :class:`EdgeDetector` из параметров конвертации (или ``None``)."""
    if not edges:
        return None
    method = edges if isinstance(edges, str) else "canny"
    return EdgeDetector(
        method=method,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        blur_ksize=blur_ksize,
    )


def _preprocess(
    frame: np.ndarray,
    *,
    detector: Optional[EdgeDetector],
    edge_mode: str,
    invert: bool,
) -> np.ndarray:
    """Применяет контуры (если включены) и инверсию к кадру перед конвертацией."""
    if detector is not None:
        frame = detector.apply(frame, mode=edge_mode)
    if invert:
        frame = 255 - frame
    return frame


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
    reverse_palette: bool = False,
    size: Optional[Tuple[int, int]] = None,
    fullcolor: bool = True,
    color_levels: Optional[int] = None,
    invert: bool = False,
    max_pixels: Optional[int] = 32_000_000,
    edges: Union[bool, str, None] = False,
    edge_mode: str = "curves",
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_ksize: int = 5,
    curve_threshold: float = 0.5,
    edge_overlay: bool = False,
    edge_fill: str = "space",
    edge_color: Optional[bool] = None,
) -> str:
    """Конвертирует изображение (путь или ``np.ndarray`` BGR) в ASCII-строку.

    Args:
        source: Путь к файлу либо уже загруженный кадр ``ndarray`` (H, W, 3) uint8.
        palette: Имя палитры (``"asii"``, ``"asii_3v"`` …) или пользовательская строка
            символов от тёмных к светлым. Используется в обычном режиме, при
            ``edge_mode="overlay"`` и как канва фона при ``edge_fill="brightness"``;
            сами контуры ВСЕГДА рисуются собственной палитрой ориентации.
        reverse_palette: Перевернуть палитру (свет/тень символами наоборот).
        size: Целевой размер ``(ширина, высота)`` в символах. ``None`` — автоподбор
            под терминал с сохранением пропорций.
        fullcolor: ``True`` — цветной ANSI truecolor, ``False`` — монохром.
        color_levels: Уровней квантования на канал при ``fullcolor=True``.
            ``None`` (по умолчанию) — цветовой охват выводится из размера палитры;
            явное число переопределяет эту зависимость. В контурных режимах
            («палитрой» считается набор символов ориентации: ``extended`` — 14
            уникальных символов → ~9 уровней на канал, ``basic`` — 4 → ~6;
            формула та же, что и для обычных палитр). При ``edge_overlay=True``
            или ``edge_fill="brightness"`` охват берётся из обычной ``palette``.
        invert: Инвертировать яркость (светлое/тёмное). Применяется, только когда
            контуры выключены либо выбран ``edge_mode="overlay"``; в чистых
            контурных режимах геометрия линий от яркости исходника не зависит.
        max_pixels: Предварительно уменьшить источник, если он больше этой площади
            (защита RAM для гигантских файлов). ``None`` — без ограничения.
        edges: Выделение контуров: ``False``/``None`` — выключено,
            ``True`` — метод по умолчанию (``"canny"``), либо строка-метод
            (``"canny"`` / ``"sobel"``). Когда включено, результат ВСЕГДА рисуется
            собственной палитрой ориентации контуров: символ повторяет наклон
            линии (``/ - \\ |``), а изогнутые участки получают парные скобки
            (``^ v < > ( ) [ ] { }``) — см. ``edge_mode``.
        edge_mode: Режим контуров (актуален только при ``edges=True``):
            ``"curves"`` (по умолчанию) — палитра ориентации + скобочные символы
            для изогнутых участков;
            ``"lines"`` — только базовый набор наклона ``/ - \\ |``, без скобок;
            ``"overlay"`` — контуры поверх оригинала с обычной яркостной
            ASCII-конвертацией (использует ``palette``/``invert``/``color_levels``).
        low_threshold / high_threshold: Пороги двойной фильтрации контуров.
        blur_ksize: Размер гауссова размытия перед детекцией (0 — выключить).
        curve_threshold: Чувствительность определения изгиба (меньше — больше
            скобочных символов). Используется в ``edge_mode="curves"``.
        edge_overlay: **Наложение контура поверх изображения** (отдельный флаг,
            по умолчанию ``False`` — НЕ накладывает). ``True`` — сначала строится
            обычная ASCII-картинка оригинала (``palette``/``invert``), затем в
            позициях контура её символы замещаются символами палитры ориентации
            (геометрия контура сохраняется). ``False`` — рисуется только сам
            контур, а фон управляется параметром ``edge_fill``.
        edge_fill: **Заполнение фона**, когда наложение ВЫКЛЮЧЕНО
            (``edge_overlay=False``; актуально для ``edge_mode="curves"|"lines"``):
            ``"space"`` (по умолчанию) — пустой фон; одиночный символ — однотонная
            канва из него (например ``"."`` или ``"#"``); ``"brightness"`` — фон
            заполняется символами яркостной ``palette`` (контуры поверх ASCII-картинки).
        edge_color: **Окрашивание контура** (отдельный флаг, ``None`` — авто:
            цвет там, где включён ``fullcolor``). ``True`` — линии контура
            получают ANSI-цвет оригинального кадра; ``False`` — контур выводится
            без цветовых кодов (даже при ``fullcolor=True``).

    Returns:
        Готовая многострочная строка ASCII-арта.
    """
    frame = _load_frame(source, max_pixels=max_pixels)

    info_w, info_h = frame.shape[1], frame.shape[0]
    w, h = _resolve_size(size, info_w, info_h, fullcolor)

    if edges:
        # Единый контурный конвейер (edges.pipeline.convert_edge_frame):
        # палитра ориентации применяется ВСЕГДА; edge_mode="overlay" —
        # предзаполнение фона яркостной канвой; edge_overlay — наложение ASCII-
        # контура ОТДЕЛЬНЫМ слоем поверх ASCII-изображения; edge_fill — выбор
        # заполнения фона при выключенном наложении.
        return convert_edge_frame(
            frame, (w, h),
            mode="extended" if edge_mode != "lines" else "basic",
            fullcolor=fullcolor, color_levels=color_levels,
            low_threshold=low_threshold, high_threshold=high_threshold,
            blur_ksize=blur_ksize, curve_threshold=curve_threshold,
            method=edges if isinstance(edges, str) else "canny",
            overlay=edge_overlay, fill=_effective_edge_fill(
                edge_mode, edge_overlay, edge_fill),
            edge_color=edge_color,
            palette=palette, reverse_palette=reverse_palette, invert=invert,
        )

    if invert:
        frame = 255 - frame

    if fullcolor:
        return frame_to_color_ansi(
            frame, palette, (w, h),
            color_levels=color_levels, reverse_palette=reverse_palette,
        )
    return frame_to_mono_text(frame, palette, (w, h), reverse_palette=reverse_palette)


def _effective_edge_fill(edge_mode: str, edge_overlay: bool, edge_fill: str) -> str:
    """Фактическое заполнение фона контурного режима.

    Legacy-режим ``edge_mode="overlay"`` означает «контуры поверх оригинала»:
    если пользователь явно не попросил послойное наложение (``edge_overlay``)
    и не выбрал свой символ заполнения, фон предзаполняется яркостной канвой.
    """
    if edge_mode == "overlay" and not edge_overlay and normalize_fill(edge_fill) == " ":
        return "brightness"
    return edge_fill


def convert_animation(
    path: str,
    *,
    palette: str = DEFAULT_PALETTE,
    reverse_palette: bool = False,
    size: Optional[Tuple[int, int]] = None,
    fullcolor: bool = True,
    color_levels: Optional[int] = None,
    invert: bool = False,
    max_pixels: Optional[int] = 32_000_000,
    progress: bool = False,
    edges: Union[bool, str, None] = False,
    edge_mode: str = "curves",
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_ksize: int = 5,
    curve_threshold: float = 0.5,
    edge_overlay: bool = False,
    edge_fill: str = "space",
    edge_color: Optional[bool] = None,
) -> Iterator[str]:
    """Конвертирует GIF/видео в **генератор** ASCII-кадров.

    Кадры читаются и конвертируются по одному (streaming), поэтому потребление
    памяти не зависит от длительности анимации — это ключевая оптимизация RAM
    по сравнению с исходным «загрузить все кадры в список».

    Args: см. :func:`convert_image` (включая ``edge_overlay`` — наложение контура
        поверх изображения отдельным флагом, и ``edge_fill`` — выбор заполнения
        фона при выключенном наложении). Дополнительно:
        progress: Показывать инкрементный прогресс-бар во время конвертации
            (если доступна библиотека ``progress``).

    Yields:
        str: очередной ASCII-кадр (ANSI, если ``fullcolor=True``).
    """
    info = probe(path)
    w, h = _resolve_size(size, info.width, info.height, fullcolor)

    frames_iter = iter_frames(path, max_pixels=max_pixels)

    # Единый контурный конвейер (как в convert_image): все edge_mode идут через
    # edges.pipeline.convert_edge_frame — корректные пропорции, наложение и
    # заполнение фона. Детекция вызывается на КАЖДЫЙ кадр (контуры движутся).
    palette_edges = bool(edges)
    edge_method = (edges if isinstance(edges, str) else "canny") if edges else "canny"
    edge_mode_out = "extended" if edge_mode != "lines" else "basic"
    fill = _effective_edge_fill(edge_mode, edge_overlay, edge_fill)
    # Охват цвета считается внутри конвейера: при наложении/яркостной канве
    # фона он привязан к обычной палитре, иначе — к палитре ориентации.

    bar = None
    if progress:
        try:
            from progress.bar import IncrementalBar

            bar = IncrementalBar("Converting", max=max(info.n_frames, 1))
        except Exception:
            bar = None

    for frame in frames_iter:
        if palette_edges:
            text = convert_edge_frame(
                frame, (w, h),
                mode=edge_mode_out,
                fullcolor=fullcolor, color_levels=color_levels,
                low_threshold=low_threshold, high_threshold=high_threshold,
                blur_ksize=blur_ksize, curve_threshold=curve_threshold,
                method=edge_method,
                overlay=edge_overlay, fill=fill, edge_color=edge_color,
                palette=palette, reverse_palette=reverse_palette, invert=invert,
            )
        else:
            base = 255 - frame if invert else frame
            if fullcolor:
                text = frame_to_color_ansi(
                    base, palette, (w, h),
                    color_levels=color_levels, reverse_palette=reverse_palette,
                )
            else:
                text = frame_to_mono_text(base, palette, (w, h), reverse_palette=reverse_palette)
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
