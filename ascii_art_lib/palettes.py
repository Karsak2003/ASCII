"""Палитры символов ASCII-арта и быстрые таблицы трансляции яркости -> символ."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

# ============================================================================
# Палитры яркости (упорядочены от «тёмных» символов к «светлым»)
# ============================================================================

ASII = "$@B%8&WM*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'."

ASII_1 = (
    "`.-':,^=;><+!rc*z?sLTvJ7FiCfI31tluneoZ5Yxjya2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"[::-1]
)

ASII_2 = "$@B%8&WM#oahkbdpqwmZO0QLCJUYXzcvunxrjf1?+~ilI;:*^\"',."

ASII_3 = (
    r" .',:`;\"i!I^lr1vjcx<>Yft*JL?T7uynozaksFVXeh3Cq2KUdp4SZbA0w5GPg9EOH6mDQNR8%&BWM#@$"
)

ASII_3V = (
    r" .',:`;\"i!I^lrvjcx<>Yft*JL?TuynozaksFVXehCqKUdpSZbAwGPgEOHmDQNR%&BWM#@$"
)

ASII_4 = r" .;coPO?@#"

#: Стандартная карта палитр по именам
PALETTES: Dict[str, str] = {
    "asii": ASII,
    "asii_1": ASII_1,
    "asii_2": ASII_2,
    "asii_3": ASII_3,
    "asii_3v": ASII_3V,
    "asii_4": ASII_4,
}


def get_palette(name_or_string: str, *, reverse: bool = False) -> str:
    """Возвращает строку-палитру по имени из :data:`PALETTES`.

    Если аргумент не является известным именем, считается, что это и есть
    пользовательская палитра, и возвращается как есть.

    Args:
        name_or_string: Имя палитры или произвольная строка символов.
        reverse: ``True`` — перевернуть палитру (порядок «светлое <-> тёмное»
            меняется на противоположный). Эквивалентно CLI-флагу ``--reverse-palette``.
    """
    if name_or_string in PALETTES:
        pal = PALETTES[name_or_string]
    else:
        pal = name_or_string
        if len(set(pal)) < 2:
            raise ValueError(f"Слишком короткая (или неизвестная) палитра: {name_or_string!r}")
    return pal[::-1] if reverse else pal


@lru_cache(maxsize=64)
def palette_color_levels(palette: str) -> int:
    """Рекомендуемое число уровней квантования цвета **по размеру палитры**.

    Цветовой охват ASCII-арта ограничен количеством различных символов, которые
    реально появляются в выводе: нет смысла красить вывод в 64 уровня на канал,
    если палитра различает всего 10 оттенков яркости. Поэтому, когда пользователь
    явно не задал ``color_levels``, мы берём величину, производную от длины
    палитры: чем богаче палитра — тем шире цветовой охват, и наоборот.

    Формула: ``round(2 + 2 * sqrt(len(palette)))`` с ограничением ``[2, 64]``
    (для стандартных палитр 10..70 символов даёт ~8..19 уровней на канал,
    т.е. ~500..7000 уникальных цветов).
    """
    n = max(2, len(set(palette)))
    levels = int(round(2 + 2.0 * (n ** 0.5)))
    return max(2, min(64, levels))


@lru_cache(maxsize=32)
def build_lut(palette: str) -> bytes:
    """Строит LUT(256) «уровень яркости 0..255 -> байт символа палитры».

    Кэшируется, поэтому повторные вызовы для одной палитры бесплатны.
    Используется вместе с ``numpy.take`` для векторизованного (без Python-циклов)
    перевода изображения в символы — главный источник прироста производительности.

    Примечание: корректно работает только для однобайтовых (ASCII) палитр;
    для палитр с многобайтовыми символами используйте :func:`is_ascii_palette`
    и fallback через индексы (см. ``converter``).
    """
    n = len(palette)
    # Таблица из 256 записей: index = яркость (0..255), value = байт символа
    table = bytearray(256)
    for level in range(256):
        idx = min(round(level / 255.0 * (n - 1)), n - 1)
        ch = palette[idx]
        try:
            table[level] = ch.encode("ascii")[0]
        except UnicodeEncodeError:
            # Неподдерживаемый символ — оставляем пробел (fallback обработается отдельно)
            table[level] = 0x20
    return bytes(table)


def is_ascii_palette(palette: str) -> bool:
    """True, если все символы палитры кодируются одним байтом ASCII."""
    try:
        palette.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


#: Палитра по умолчанию (оптимизированная «светлая» версия asii_3)
DEFAULT_PALETTE = ASII_3V


# ============================================================================
# Standard Galactic Alphabet (дополнительный набор символов)
# ============================================================================

SGA: Dict[str, str] = {
    "a": "ᔑ", "b": "ʖ", "c": "ᓵ", "d": "↸", "e": "ᒷ", "f": "⎓",
    "g": "⊣", "h": "⍑", "я": "╎", "j": "⋮", "k": "ꖌ", "l": "ꖎ",
    "m": "ᒲ", "n": "リ", "o": "𝙹", "p": "⇅", "q": "ᑑ", "r": "∷",
    "s": "ᓭ", "t": "ℸ", "u": "⚍", "v": "⍊", "w": "∴", "x": "/",
    "y": "|", "z": "⨅",
}
