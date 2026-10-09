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

from .converter import frame_to_color_ansi, frame_to_mono_text, frame_to_symbol_bytes
from .edge_palette import (
    apply_edge_fill,
    frame_to_edge_symbols,
    get_edge_palette,
    normalize_fill,
)
from .edges import EdgeDetector
from .media import classify, iter_frames, probe
from .palettes import DEFAULT_PALETTE, get_palette, palette_color_levels
from .renderer import ConsoleRenderer

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

    if edges and edge_mode in ("curves", "lines"):
        info_w, info_h = frame.shape[1], frame.shape[0]
        w, h = _resolve_size(size, info_w, info_h, fullcolor)
        method = edges if isinstance(edges, str) else "canny"
        return _edge_palette_text(
            frame, (w, h),
            mode="extended" if edge_mode == "curves" else "basic",
            fullcolor=fullcolor, color_levels=color_levels,
            low_threshold=low_threshold, high_threshold=high_threshold,
            blur_ksize=blur_ksize, curve_threshold=curve_threshold, method=method,
            overlay=edge_overlay, fill=edge_fill, edge_color=edge_color,
            palette=palette, reverse_palette=reverse_palette, invert=invert,
        )

    if edges and edge_mode == "overlay" and (edge_overlay or edge_fill != "space"):
        # Старый режим «контуры поверх оригинала» с яркостной палитрой +
        # новые независимые флаги: наложение контура и заполнение фона.
        method = edges if isinstance(edges, str) else "canny"
        info_w, info_h = frame.shape[1], frame.shape[0]
        w, h = _resolve_size(size, info_w, info_h, fullcolor)
        base = 255 - frame if invert else frame
        if fullcolor:
            text = frame_to_color_ansi(base, palette, (w, h),
                                       color_levels=color_levels,
                                       reverse_palette=reverse_palette)
        else:
            text = frame_to_mono_text(base, palette, (w, h),
                                      reverse_palette=reverse_palette)
        if not edge_overlay:
            return text
        sym, line_mask = _edge_symbol_maps(
            frame, (w, h), mode="extended",
            low_threshold=low_threshold, high_threshold=high_threshold,
            blur_ksize=blur_ksize, curve_threshold=curve_threshold, method=method)
        return _overlay_edges_on_text(text, line_mask, sym,
                                      sym.shape[1] != w)

    detector = _make_detector(
        edges,
        edge_mode=edge_mode,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        blur_ksize=blur_ksize,
    )
    frame = _preprocess(frame, detector=detector, edge_mode=edge_mode, invert=invert)

    info_w, info_h = frame.shape[1], frame.shape[0]
    w, h = _resolve_size(size, info_w, info_h, fullcolor)

    if fullcolor:
        return frame_to_color_ansi(
            frame, palette, (w, h),
            color_levels=color_levels, reverse_palette=reverse_palette,
        )
    return frame_to_mono_text(frame, palette, (w, h), reverse_palette=reverse_palette)


def _edge_canvas_ansi(canvas_u8: np.ndarray, frame_rgb: np.ndarray, levels: int) -> str:
    """ANSI-сборка яркостной ASCII-канвы (H, W) uint8 с квантованным цветом кадра.

    Используется для **окрашивания контура** (``edge_color=True``): позиции линий
    уже замещены символами палитры ориентации, остальной фон — символы ``palette``.
    Печатаются все позиции строки (пробелы сохраняются посимвольно), ANSI-префикс
    ставится только при смене квантованного цвета (дельта-кодирование).
    """
    h, w = canvas_u8.shape
    if frame_rgb.shape[0] != h or frame_rgb.shape[1] != w:
        import cv2 as _cv2
        frame_rgb = _cv2.resize(frame_rgb, (int(w), int(h)), interpolation=_cv2.INTER_AREA)
    if frame_rgb.ndim == 2:
        frame_rgb = np.dstack([frame_rgb] * 3)

    q = max(1, 256 // max(1, min(256, int(levels))))
    lvl = (frame_rgb.astype(np.uint16) + (q // 2)) // q
    lvl = np.clip(lvl, 0, levels - 1)
    val = (lvl * q + (q // 2)).clip(0, 255).astype(np.uint8)
    cid = ((val[:, :, 2].astype(np.int32) << 16)
           | (val[:, :, 1].astype(np.int32) << 8)
           | val[:, :, 0].astype(np.int32))

    change = np.empty(cid.shape, dtype=bool)
    change[:, 0] = True
    change[:, 1:] = cid[:, 1:] != cid[:, :-1]

    uniq_ids = np.unique(cid[change])
    prefix_at = dict(
        zip(
            uniq_ids.tolist(),
            (f"\033[38;2;{(i >> 16) & 0xFF};{(i >> 8) & 0xFF};{i & 0xFF}m"
             for i in uniq_ids.tolist()),
        )
    )

    text_rows = [r.tobytes().decode("ascii") for r in canvas_u8]
    cid_list = cid.tolist()
    chg_list = change.tolist()
    out_lines = []
    for y in range(h):
        row_sym = text_rows[y]
        row_cid = cid_list[y]
        row_ch = chg_list[y]
        parts = []
        run = ""
        for x, ch in enumerate(row_sym):
            if row_ch[x]:
                if run:
                    parts.append(run)
                run = prefix_at[row_cid[x]]
            run += ch
        if run:
            parts.append(run)
        parts.append("\033[0m")
        out_lines.append("".join(parts))
    return "\n".join(out_lines)


def _overlay_edges_on_text(text: str, line_mask: np.ndarray, sym: np.ndarray,
                           unicode_grid: bool) -> str:
    """Послойно встраивает символы палитры ориентации в готовый ANSI/текст.

    Универсальный путь наложения контура поверх изображения: работает с любым
    выводом яркостной конвертации (включая Unicode-палитры и truecolor-ANSI).
    Символ в позиции линии замещается символом из кодовой сетки ``sym``
    (escape-пары разворачиваются), ANSI-префиксы строки сохраняются.
    """
    from .edge_palette import decode_grid

    sym_rows = decode_grid(sym)
    lines = text.split("\n")
    out_lines = []
    for y in range(len(lines)):
        line = lines[y]
        # Разбор строки на токены: escape-последовательности и печатные символы
        cells = []  # (start, end) каждого символа-позиции
        i = 0
        n = len(line)
        while i < n:
            if line[i] == "\x1b":
                j = i + 1
                while j < n and not ("@" <= line[j] <= "~"):  # финальный байт CSI
                    j += 1
                i = j + 1 if j < n else n
                continue
            cp = ord(line[i])
            step = 2 if (0xD800 <= cp <= 0xDBFF and i + 1 < n) else 1
            cells.append((i, i + step))
            i += step
        row = list(line)
        srow = sym_rows[y] if y < len(sym_rows) else ""
        mask = line_mask[y]
        x = 0
        for (a, b) in cells:
            if x < mask.shape[0] and mask[x]:
                if x < len(srow):
                    ch = srow[x]
                    if (a + 1 < b) != (len(ch) == 2):
                        break  # нестандартный шрифт внутри палитры — без замещения
                    row[a:b] = list(ch)
            x += 1
        out_lines.append("".join(row))
    remaining = lines[len(out_lines):]
    return "\n".join(out_lines + remaining)


def _edge_palette_text(
    frame: np.ndarray,
    size: Tuple[int, int],
    *,
    mode: str = "extended",
    fullcolor: bool,
    color_levels: Optional[int],
    low_threshold: int,
    high_threshold: int,
    blur_ksize: int,
    curve_threshold: float,
    method: str,
    overlay: bool = False,
    fill: str = "space",
    edge_color: Optional[bool] = None,
    palette: str = DEFAULT_PALETTE,
    reverse_palette: bool = False,
    invert: bool = False,
) -> str:
    """Монохромный/цветной вывод собственной палитрой ориентации контуров.

    Args:
        mode: ``"extended"`` — наклон + скобки для изгибов; ``"basic"`` — только
            ``/ - \\ |``.
        color_levels: ``None`` — охват выводится из размера самой палитры
            ориентации (:func:`palette_color_levels` по уникальным символам);
            явное число переопределяет эту зависимость. Если же включено
            наложение (``overlay=True``) или яркостная канва фона
            (``fill="brightness"``), вывод занимает и обычная палитра — тогда
            охват берётся из неё.
        overlay: **Наложение контура поверх изображения** (по умолчанию ``False``
            — НЕ накладывает): сначала строится яркостная ASCII-картинка кадра
            (с учётом ``invert``/``reverse_palette``), затем в позициях линий её
            символы замещаются символами палитры ориентации.
        fill: заполнение фона при ``overlay=False``: ``"space"`` — пусто,
            одиночный символ (напр. ``"."``) — однотонная канва,
            ``"brightness"`` — яркостные символы ``palette`` под контуром.
        edge_color: **Окрашивание контура** (``None`` — авто: цвет там, где
            включён ``fullcolor``). ``True`` — позиции контура получают ANSI-цвет
            оригинального кадра (квантованный ``color_levels``); ``False`` —
            контур выводится без цветовых кодов даже при ``fullcolor=True``.
    """
    from .edge_palette import edge_symbols_to_text

    common = dict(
        mode=mode,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        blur_ksize=blur_ksize,
        curve_threshold=curve_threshold,
        method=method,
    )
    f = normalize_fill(fill)
    needs_canvas = bool(overlay) or f == "brightness"
    colored_edge = fullcolor if edge_color is None else bool(edge_color)

    # Охват цвета: палитра ориентации, если канва не участвует; иначе — обычная.
    eff_levels = color_levels
    if eff_levels is None:
        pal_for_levels = (get_palette(palette, reverse=reverse_palette)
                          if needs_canvas else get_edge_palette(mode))
        eff_levels = palette_color_levels(pal_for_levels)

    sym, line_mask = _edge_symbol_maps(frame, size, **common)
    unicode_grid = sym.shape[1] != size[0]

    # --- Наложение контура поверх изображения (отдельный флаг, по умолчанию
    # ВЫКЛЮЧЕНО): контур встраивается ОТДЕЛЬНО поверх ASCII-версии оригинала. ---
    if overlay:
        canvas_u8 = None
        if not colored_edge:
            src = 255 - frame if invert else frame
            canvas_u8 = frame_to_symbol_bytes(src, palette, size,
                                              reverse_palette=reverse_palette)
        if canvas_u8 is not None:
            # Быстрый байтовый путь: замещение позиций линий + ANSI-сборка
            sym = _overlay_edges_on_canvas(sym, canvas_u8, line_mask)
            if colored_edge:
                return _edge_canvas_ansi(sym, frame, int(eff_levels))
            return edge_symbols_to_text(sym)
        # Универсальный путь (цветной вывод / Unicode-палитры): сначала обычная
        # яркостная конвертация, затем послойное замещение символов контура.
        base = 255 - frame if invert else frame
        if colored_edge:
            text = frame_to_color_ansi(base, palette, size,
                                       color_levels=color_levels,
                                       reverse_palette=reverse_palette)
        else:
            text = frame_to_mono_text(base, palette, size,
                                      reverse_palette=reverse_palette)
        return _overlay_edges_on_text(text, line_mask, sym, unicode_grid)

    # --- Без наложения: управляемое заполнение фона --------------------------
    if f != " ":
        canvas_u8 = None
        if f == "brightness":
            src = 255 - frame if invert else frame
            canvas_u8 = frame_to_symbol_bytes(src, palette, size,
                                              reverse_palette=reverse_palette)
            if canvas_u8 is None:
                f = " "  # Unicode-палитра — яркостная канва недоступна, фон пустой
        if f != " ":
            sym = apply_edge_fill(sym, line_mask, f,
                                  unicode_grid=unicode_grid,
                                  brightness_bytes=canvas_u8)
            if f == "brightness" and colored_edge:
                # Яркостная канва участвует в выводе — красим весь кадр цветом
                # оригинала (позиции линий уже содержат символы ориентации).
                return _edge_canvas_ansi(sym, frame, int(eff_levels))

    if colored_edge:
        return _edge_ansi_from_symbols(sym, frame, int(eff_levels))
    return edge_symbols_to_text(sym)


def _edge_symbol_maps(frame: np.ndarray, size: Tuple[int, int], **common):
    """(кодовая сетка символов, маска линий) — единый проход детекции."""
    from .edge_palette import _edge_maps

    m = _edge_maps(frame, size, **common)
    return m["grid"], m["line_mask"]


def _overlay_edges_on_canvas(edge_grid: np.ndarray, canvas_u8: np.ndarray,
                             line_mask: np.ndarray) -> np.ndarray:
    """Встраивает кодовую сетку контуров в яркостную ASCII-канву (H, W).

    Позиции линий замещаются символами палитры ориентации (escape-пары
    сохраняются), остальные позиции остаются символами обычной палитры.
    """
    h, w = canvas_u8.shape
    out = np.full((h, w * 2), 0x20, dtype=np.uint8)
    out[:, 0::2] = canvas_u8
    eg = edge_grid
    if eg.shape[1] == w * 2:                       # extended: пары байтов выровнены
        sel = line_mask[:, 0::2]
        out[sel, ::2] = eg[:, 0::2][sel]
        out[sel, 1::2] = eg[:, 1::2][sel]
    else:                                          # basic: одиночные байты
        out[line_mask, 0::2] = eg[line_mask]
    return out

def _edge_ansi_from_symbols(sym: np.ndarray, frame_rgb: np.ndarray, levels: int) -> str:
    """ANSI-сборка контурного ASCII из готовой карты символов."""
    h = sym.shape[0]
    unicode_grid = sym.ndim == 2 and sym.shape[1] != frame_rgb.shape[1]
    w = sym.shape[1] // 2 if unicode_grid else sym.shape[1]
    
    # Приводим размер кадра к размеру карты символов
    if frame_rgb.shape[0] != h or frame_rgb.shape[1] != w:
        import cv2 as _cv2
        frame_rgb = _cv2.resize(frame_rgb, (int(w), int(h)), interpolation=_cv2.INTER_AREA)
    if frame_rgb.ndim == 2:
        frame_rgb = np.dstack([frame_rgb] * 3)
        
    q = max(1, 256 // max(1, min(256, int(levels))))
    lvl = (frame_rgb.astype(np.uint16) + (q // 2)) // q
    lvl = np.clip(lvl, 0, levels - 1)
    val = (lvl * q + (q // 2)).clip(0, 255).astype(np.uint8)
    
    cid = ((val[:, :, 2].astype(np.int32) << 16)
           | (val[:, :, 1].astype(np.int32) << 8)
           | val[:, :, 0].astype(np.int32))
           
    change = np.empty(cid.shape, dtype=bool)
    change[:, 0] = True
    change[:, 1:] = cid[:, 1:] != cid[:, :-1]
    
    uniq_ids = np.unique(cid[change])
    prefix_at = dict(
        zip(
            uniq_ids.tolist(),
            (f"\033[38;2;{(i >> 16) & 0xFF};{(i >> 8) & 0xFF};{i & 0xFF}m" for i in uniq_ids.tolist()),
        )
    )
    
    # Импортируем словарь для расшифровки ESC-последовательностей
    from .edge_palette import _EXT_CODE
    
    cid_list = cid.tolist()
    chg_list = change.tolist()
    out_lines = []
    
    for y in range(h):
        row_cid = cid_list[y]
        row_ch = chg_list[y]
        parts = []
        run = ""
        
        # Итерируемся строго по ширине исходной сетки (w), а не по длине decoded-строки
        for x in range(w):
            # Получаем символ из кодовой сетки
            if unicode_grid:
                b1 = int(sym[y, 2 * x])
                b2 = int(sym[y, 2 * x + 1])
                if b1 == 0x1B:
                    ch = _EXT_CODE.get(b2, "?")
                else:
                    ch = chr(b1)
            else:
                ch = chr(int(sym[y, x]))
            
            # Меняем ANSI-префикс только при смене цвета
            if row_ch[x]:
                if run:
                    parts.append(run)
                run = prefix_at[row_cid[x]]
            
            # Символ добавляется ВСЕГДА (включая пробелы-заполнители)
            run += ch
                
        if run:
            parts.append(run)
        parts.append("\033[0m")
        out_lines.append("".join(parts))
        
    return "\n".join(out_lines)

# ---------------------------------------------------------------------------
# Конвертация анимации (GIF / видео) — ленивый генератор, экономия RAM
# ---------------------------------------------------------------------------

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

    palette_edges = bool(edges) and edge_mode in ("curves", "lines")
    method = (edges if isinstance(edges, str) else "canny") if palette_edges else "canny"
    edge_common = dict(
        mode="extended" if edge_mode == "curves" else "basic",
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        blur_ksize=blur_ksize,
        curve_threshold=curve_threshold,
        method=method,
    )
    # Охват цвета считается внутри _edge_palette_text: при наложении/яркостной
    # канве фона он привязан к обычной палитре, иначе — к палитре ориентации.

    detector = None
    if edges and not palette_edges:
        detector = _make_detector(
            edges,
            edge_mode=edge_mode,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            blur_ksize=blur_ksize,
        )

    bar = None
    if progress:
        try:
            from progress.bar import IncrementalBar

            bar = IncrementalBar("Converting", max=max(info.n_frames, 1))
        except Exception:
            bar = None

    for frame in frames_iter:
        if palette_edges:
            text = _edge_palette_text(
                frame, (w, h),
                mode=edge_common["mode"],
                fullcolor=fullcolor, color_levels=color_levels,
                low_threshold=low_threshold, high_threshold=high_threshold,
                blur_ksize=blur_ksize, curve_threshold=curve_threshold,
                method=method,
                overlay=edge_overlay, fill=edge_fill, edge_color=edge_color,
                palette=palette, reverse_palette=reverse_palette, invert=invert,
            )
        else:
            base = _preprocess(frame, detector=detector, edge_mode=edge_mode,
                               invert=invert)
            if fullcolor:
                text = frame_to_color_ansi(
                    base, palette, (w, h),
                    color_levels=color_levels, reverse_palette=reverse_palette,
                )
            else:
                text = frame_to_mono_text(base, palette, (w, h), reverse_palette=reverse_palette)
            # Старый режим edge_mode="overlay" + независимые флаги: наложение
            # контура отдельным слоем поверх яркостной ASCII-картинки.
            if (edges and edge_mode == "overlay" and edge_overlay):
                sym, line_mask = _edge_symbol_maps(
                    frame, (w, h), mode="extended",
                    low_threshold=low_threshold, high_threshold=high_threshold,
                    blur_ksize=blur_ksize, curve_threshold=curve_threshold,
                    method=(edges if isinstance(edges, str) else "canny"))
                text = _overlay_edges_on_text(text, line_mask, sym,
                                              sym.shape[1] != w)
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
