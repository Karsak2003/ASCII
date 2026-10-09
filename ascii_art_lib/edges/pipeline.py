"""Высокоуровневая сборка контурного вывода (палитра ориентации -> текст/ANSI).

Модуль инкапсулирует «кухню» режимов ``edges=True``: единый проход детекции
(:func:`~ascii_art_lib.edges.palette._edge_maps`), разворот компактной сетки в
полноширинную, яркостная канва для наложения/заполнения и ANSI-подсветка через
:class:`~ascii_art_lib.utils.ansi.TruecolorHighlighter`. Всё это было размазано
по ``api.py`` дублирующимися ветками; здесь — одна точка сборки, используемая
и статичной конвертацией, и пофреймовой конвертацией анимаций.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ascii_art_lib.core.converter import frame_to_symbol_bytes
from ascii_art_lib.core.image_ops import resize_frame
from ascii_art_lib.core.palettes import DEFAULT_PALETTE, get_palette, palette_color_levels
from ascii_art_lib.utils.ansi import TruecolorHighlighter
from ascii_art_lib.utils.text import grid_rows, rows_to_text

from .palette import (
    _edge_maps,
    apply_edge_fill,
    decode_grid,
    expand_edge_grid,
    get_edge_palette,
    normalize_fill,
)

__all__ = ["convert_edge_frame", "overlay_edges_on_text"]


def _resolve_levels(
    color_levels: Optional[int],
    *,
    mode: str,
    needs_canvas: bool,
    palette: str,
    reverse_palette: bool,
) -> int:
    """Охват цвета: из палитры ориентации (или обычной, если канва участвует)."""
    if color_levels is not None:
        return max(1, min(256, int(color_levels)))
    pal = (get_palette(palette, reverse=reverse_palette)
           if needs_canvas else get_edge_palette(mode))
    return palette_color_levels(pal)


def overlay_edges_on_text(text: str, line_mask: np.ndarray, sym: np.ndarray,
                          unicode_grid: bool) -> str:
    """Послойно встраивает символы палитры ориентации в готовый ANSI/текст.

    Резервный путь наложения контура поверх изображения для Unicode-палитр
    (байтовая канва невозможна): работает с любым выводом яркостной конвертации
    (включая truecolor-ANSI). Символ в позиции линии замещается символом из
    кодовой сетки ``sym`` (escape-пары разворачиваются), ANSI-префиксы строк
    сохраняются.
    """
    sym_rows = decode_grid(sym)
    lines_t = text.split("\n")
    out_lines: List[str] = []
    for y in range(len(lines_t)):
        line = lines_t[y]
        # Разбор строки на токены: escape-последовательности и печатные символы
        cells: List[Tuple[int, int]] = []  # (start, end) каждого символа-позиции
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
    remaining = lines_t[len(out_lines):]
    return "\n".join(out_lines + remaining)


def convert_edge_frame(
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
    """Кадр -> ASCII-текст собственной палитрой ориентации контуров.

    Единая точка сборки всех контурных режимов (используется ``api.convert_image``
    и ``api.convert_animation``):

    * ``mode``: ``"extended"`` — наклон + скобки для изгибов; ``"basic"`` —
      только ``/ - \\ |``;
    * ``overlay``: **наложение ASCII-контура поверх ASCII-изображения**
      (по умолчанию ``False`` — НЕ накладывает);
    * ``fill``: заполнение фона при выключенном наложении (``"space"``,
      одиночный символ, ``"brightness"``);
    * ``edge_color``: окрашивание контура (``None`` — авто от ``fullcolor``).

    Все пути сборки работают над **полноширинной** кодовой сеткой ``(h, w)``
    (см. :func:`expand_edge_grid`) — ровно один текстовый символ на ячейку
    изображения, поэтому пропорции картинки никогда не искажаются.
    """
    maps = _edge_maps(
        frame, size,
        mode=mode, low_threshold=low_threshold, high_threshold=high_threshold,
        blur_ksize=blur_ksize, curve_threshold=curve_threshold, method=method,
    )
    sym: np.ndarray = maps["grid"]
    line_mask: np.ndarray = maps["line_mask"]

    f = normalize_fill(fill)
    needs_canvas = bool(overlay) or f == "brightness"
    colored_edge = fullcolor if edge_color is None else bool(edge_color)
    eff_levels = _resolve_levels(color_levels, mode=mode, needs_canvas=needs_canvas,
                                 palette=palette, reverse_palette=reverse_palette)

    # Компактная escape-сетка extended-палитры имеет ширину 2*w — отличаем её
    # именно по ширине (одиночные ASCII в extended тоже хранятся дублями).
    unicode_grid = sym.ndim == 2 and sym.shape[1] == 2 * size[0]

    # Цвет привязан к колонкам исходной сетки: для escape-пар — обе позиции пары.
    color_cols = np.column_stack([line_mask[:, 0::2]] * 2) if unicode_grid else line_mask
    sym_w = expand_edge_grid(sym, size[0]) if unicode_grid else sym
    mask_full = color_cols if unicode_grid else line_mask

    # --- Яркостная канва (фон "brightness" либо наложение контура) ------------
    canvas_u8: Optional[np.ndarray] = None
    if needs_canvas:
        src = 255 - frame if invert else frame
        canvas_u8 = frame_to_symbol_bytes(src, palette, size,
                                          reverse_palette=reverse_palette)
        if canvas_u8 is None:
            # Unicode-палитра: байтовая канва недоступна.
            if overlay:
                from ascii_art_lib.core.converter import (
                    frame_to_color_ansi,
                    frame_to_mono_text,
                )

                base = 255 - frame if invert else frame
                if fullcolor and colored_edge:
                    text = frame_to_color_ansi(base, palette, size,
                                               color_levels=color_levels,
                                               reverse_palette=reverse_palette)
                else:
                    text = frame_to_mono_text(base, palette, size,
                                              reverse_palette=reverse_palette)
                return overlay_edges_on_text(text, line_mask, sym, unicode_grid)
            f = " "
            needs_canvas = False

    if needs_canvas and canvas_u8 is not None:
        cw = int(size[0])
        mask = line_mask[:, :cw] if line_mask.shape[1] >= cw else line_mask
        if overlay:
            # НАЛОЖЕНИЕ: базовый слой — яркостная ASCII-канва оригинала; в
            # позициях линий она ЗАМЕЩАЕТСЯ символом палитры ориентации.
            # Пользовательский fill к наложению не применяется (иначе затирался
            # бы рисунок базовой картинки).
            grid_full = sym_w.copy()
            grid_full[~mask] = canvas_u8[~mask]
        else:
            # Только заполнение фона яркостью (позиции линий сохраняются).
            bg_compact = apply_edge_fill(sym, line_mask, f, unicode_grid=unicode_grid,
                                         brightness_bytes=None)
            bg_full = expand_edge_grid(bg_compact, cw) if unicode_grid else bg_compact
            grid_full = bg_full.copy()
            empty = (grid_full == 0x20) & ~mask
            grid_full[empty] = canvas_u8[empty]

        rows = grid_rows(grid_full)
        if fullcolor and colored_edge:
            return TruecolorHighlighter(eff_levels).apply(rows, frame)
        return rows_to_text(rows)

    # --- Без канвы: символ заполнения или пустой фон --------------------------
    if f != " ":
        filled = apply_edge_fill(sym, line_mask, f, unicode_grid=unicode_grid)
        sym_w = expand_edge_grid(filled, size[0]) if unicode_grid else filled

    if fullcolor and colored_edge:
        # Красим ТОЛЬКО позиции контура (фон остаётся неокрашенным)
        hl = TruecolorHighlighter(eff_levels, color_mask=mask_full, edge_only=True)
        return hl.apply(grid_rows(sym_w), frame)
    return rows_to_text(grid_rows(sym_w))
