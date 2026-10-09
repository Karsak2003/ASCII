"""Подпакет выделения контуров: детекция + палитра ориентации.

* :mod:`~ascii_art_lib.edges.detector` — pipeline Canny/Sobel (blur -> Sobel ->
  magnitude -> двойной порог с гистерезисом), класс :class:`EdgeDetector`;
* :mod:`~ascii_art_lib.edges.palette` — палитра ориентации контуров: символ
  зависит от наклона/кривизны линии (``/ - \\ |``, изгибы ``^ v < > ( ) [ ] { }``,
  стыки ``+ x``, стрелки, двойные линии), кодовая uint8-сетка и её разворот в
  полноширинный вывод;
* :mod:`~ascii_art_lib.edges.pipeline` — высокоуровневая сборка контурного
  вывода (монохром / ANSI / наложение контура поверх ASCII-изображения /
  заполнение фона) поверх этих двух модулей.
"""

from __future__ import annotations

from .detector import EdgeDetector, blend_with_source, detect_edges
from .palette import (
    EDGE_BASIC,
    EDGE_EXTENDED,
    EDGE_FILL_PRESETS,
    EDGE_FILLS,
    EDGE_PALETTES,
    apply_edge_fill,
    decode_grid,
    edge_palette_symbols,
    edge_symbols_to_text,
    expand_edge_grid,
    frame_to_edge_ansi,
    frame_to_edge_symbols,
    get_edge_palette,
    grid_to_text,
    is_edge_fill_valid,
    normalize_fill,
)
from .pipeline import convert_edge_frame, overlay_edges_on_text

__all__ = [
    "EdgeDetector",
    "detect_edges",
    "blend_with_source",
    "EDGE_BASIC",
    "EDGE_EXTENDED",
    "EDGE_PALETTES",
    "EDGE_FILLS",
    "EDGE_FILL_PRESETS",
    "get_edge_palette",
    "edge_palette_symbols",
    "frame_to_edge_symbols",
    "edge_symbols_to_text",
    "frame_to_edge_ansi",
    "normalize_fill",
    "is_edge_fill_valid",
    "apply_edge_fill",
    "decode_grid",
    "grid_to_text",
    "expand_edge_grid",
    "convert_edge_frame",
    "overlay_edges_on_text",
]
