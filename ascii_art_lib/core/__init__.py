"""Ядро пакета: низкоуровневые конвертеры, палитры и работа с медиа.

Модули уровня ``core`` не знают ничего о контурах, CLI или высокоуровневом
API — это «строительные блоки»:

* :mod:`~ascii_art_lib.core.palettes` — палитры символов яркости + LUT;
* :mod:`~ascii_art_lib.core.threshold_map` — матрицы дизеринга и ядер фильтров;
* :mod:`~ascii_art_lib.core.image_ops` — общие векторизованные операции над
  изображениями (resize/cvtColor) — переиспользуются всеми модулями;
* :mod:`~ascii_art_lib.core.media` — ленивое пофреймовое чтение файлов;
* :mod:`~ascii_art_lib.core.converter` — яркостная конвертация кадра в ASCII.
"""

from __future__ import annotations

from .converter import (
    frame_to_color_ansi,
    frame_to_mono_text,
    frame_to_symbol_bytes,
    frame_to_symbols,
    quantize,
    symbols_to_text,
)
from .image_ops import resize_frame, to_bgr, to_gray
from .media import MediaInfo, classify, get_fps, iter_frames, probe
from .palettes import (
    DEFAULT_PALETTE,
    PALETTES,
    build_lut,
    get_palette,
    is_ascii_palette,
    palette_color_levels,
)
from .threshold_map import ThresholdMap

__all__ = [
    "frame_to_symbols",
    "frame_to_symbol_bytes",
    "symbols_to_text",
    "frame_to_color_ansi",
    "frame_to_mono_text",
    "quantize",
    "to_gray",
    "to_bgr",
    "resize_frame",
    "MediaInfo",
    "classify",
    "iter_frames",
    "probe",
    "get_fps",
    "DEFAULT_PALETTE",
    "PALETTES",
    "get_palette",
    "build_lut",
    "is_ascii_palette",
    "palette_color_levels",
    "ThresholdMap",
]
