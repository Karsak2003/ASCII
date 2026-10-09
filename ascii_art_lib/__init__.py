"""ascii_art_lib — быстрая конвертация изображений/анимаций в ASCII-арт.

Пакет предоставляет:

* высокоуровневые функции :func:`convert_image`, :func:`convert_animation`,
  :func:`show_image`, :func:`play_animation`, :func:`save_ascii`;
* класс вывода в консоль :class:`ConsoleRenderer`;
* низкоуровневые векторизованные конвертеры (:mod:`ascii_art_lib.core.converter`);
* ленивое чтение медиафайлов (:mod:`ascii_art_lib.core.media`);
* палитры символов (:mod:`ascii_art_lib.core.palettes`).

Пример::

    from ascii_art_lib import convert_image, play_animation

    art = convert_image("photo.jpg", size=(120, 40), fullcolor=True)
    print(art)

    for frame in play_animation("clip.gif"):   # генератор — память O(1 кадра)
        ...
"""

from __future__ import annotations

from .api import (
    convert_animation,
    convert_image,
    play_animation,
    save_ascii,
    show_image,
    terminal_fit_size,
)
from .core.converter import (
    frame_to_color_ansi,
    frame_to_mono_text,
    frame_to_symbols,
    symbols_to_text,
)
from .core.media import MediaInfo, classify, iter_frames, probe
from .core.palettes import (
    ASII,
    ASII_1,
    ASII_2,
    ASII_3,
    ASII_3V,
    ASII_4,
    DEFAULT_PALETTE,
    PALETTES,
    SGA,
    get_palette,
    palette_color_levels,
)
from .core.threshold_map import ThresholdMap
from .edges import (
    EdgeDetector,
    blend_with_source,
    detect_edges,
)
from .edges.palette import (
    EDGE_BASIC,
    EDGE_EXTENDED,
    EDGE_PALETTES,
    EDGE_FILLS,
    EDGE_FILL_PRESETS,
    apply_edge_fill,
    is_edge_fill_valid,
    normalize_fill,
    edge_palette_symbols,
    edge_symbols_to_text,
    frame_to_edge_ansi,
    frame_to_edge_symbols,
    get_edge_palette,
)
from .rendering import ConsoleRenderer

__version__ = "1.0.0"

__all__ = [
    # API
    "convert_image",
    "convert_animation",
    "show_image",
    "play_animation",
    "save_ascii",
    "terminal_fit_size",
    # Конвертеры
    "frame_to_symbols",
    "symbols_to_text",
    "frame_to_color_ansi",
    "frame_to_mono_text",
    # Контурная детекция
    "EdgeDetector",
    "detect_edges",
    "blend_with_source",
    # Палитра ориентации контуров
    "EDGE_BASIC",
    "EDGE_EXTENDED",
    "EDGE_PALETTES",
    "get_edge_palette",
    "edge_palette_symbols",
    "frame_to_edge_symbols",
    "edge_symbols_to_text",
    "frame_to_edge_ansi",
    # Заполнение фона контуров / наложение контура
    "EDGE_FILLS",
    "EDGE_FILL_PRESETS",
    "normalize_fill",
    "is_edge_fill_valid",
    "apply_edge_fill",
    # Медиа
    "MediaInfo",
    "classify",
    "iter_frames",
    "probe",
    # Рендер
    "ConsoleRenderer",
    # Палитры
    "ASII", "ASII_1", "ASII_2", "ASII_3", "ASII_3V", "ASII_4",
    "DEFAULT_PALETTE", "PALETTES", "SGA", "get_palette", "palette_color_levels",
    # Прочее
    "ThresholdMap",
    "__version__",
]
