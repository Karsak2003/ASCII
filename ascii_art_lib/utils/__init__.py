"""Общие низкоуровневые утилиты пакетa ascii_art_lib.

* :mod:`~ascii_art_lib.utils.ansi` — абстракции подсветки (ANSI-цвет) и рендера;
* :mod:`~ascii_art_lib.utils.text` — сборка текстовых строк из кодовых сеток.
"""

from __future__ import annotations

from .ansi import (
    ANSI_RESET,
    BaseRenderer,
    NoHighlighter,
    NullRenderer,
    SyntaxHighlighter,
    TruecolorHighlighter,
    ansi_fg_truecolor,
    build_prefix_map,
    delta_change_mask,
    pack_color_id,
    quantize_color,
    strip_ansi,
)
from .text import grid_rows, rows_to_text

__all__ = [
    "ANSI_RESET",
    "ansi_fg_truecolor",
    "strip_ansi",
    "quantize_color",
    "pack_color_id",
    "delta_change_mask",
    "build_prefix_map",
    "SyntaxHighlighter",
    "NoHighlighter",
    "TruecolorHighlighter",
    "BaseRenderer",
    "NullRenderer",
    "grid_rows",
    "rows_to_text",
]
