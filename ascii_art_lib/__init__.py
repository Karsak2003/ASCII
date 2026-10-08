"""ascii_art_lib — быстрая конвертация изображений/анимаций в ASCII-арт.

Пакет предоставляет:

* высокоуровневые функции :func:`convert_image`, :func:`convert_animation`,
  :func:`show_image`, :func:`play_animation`, :func:`save_ascii`;
* класс вывода в консоль :class:`ConsoleRenderer`;
* низкоуровневые векторизованные конвертеры (:mod:`ascii_art_lib.converter`);
* ленивое чтение медиафайлов (:mod:`ascii_art_lib.media`);
* палитры символов (:mod:`ascii_art_lib.palettes`).

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
from .converter import (
    frame_to_color_ansi,
    frame_to_mono_text,
    frame_to_symbols,
    symbols_to_text,
)
from .edges import EdgeDetector, blend_with_source, detect_edges
from .media import MediaInfo, classify, iter_frames, probe
from .palettes import (
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
from .renderer import ConsoleRenderer
from .threshold_map import ThresholdMap

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
