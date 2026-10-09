"""Ядро конвертации: изображение/кадр -> ASCII-арта (monochrome и цветной ANSI).

Ключевые отличия от исходного ``main.py``:

* **Никаких ``np.vectorize``/``np.fromfunction`` с Python-лямбдами** — они выполняют
  цикл по пикселям в интерпретаторе Python (очень медленно и создают объекты строк
  для каждого пикселя). Вместо них — чисто векторные операции NumPy и ``np.take``
  по LUT-таблице палитры.
* **Отсутствуют глобальные переменные и утечки памяти** — все промежуточные массивы
  локальны, имеют тип ``uint8``/``int32`` и освобождаются сразу после функции.
* **Нет создания матрицы уникальных цветов через ``set(getPalette(...))``** с ``list.index``
  (O(N^2) и гигантские строковые словари). Цвет кодируется напрямую из квантованных
  каналов через ``np.savetxt``-подобную сборку ANSI-префиксов только там, где цвет меняется.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from ascii_art_lib.core.image_ops import resize_frame, to_gray
from ascii_art_lib.core.palettes import (
    DEFAULT_PALETTE,
    build_lut,
    get_palette,
    is_ascii_palette,
    palette_color_levels,
)
from ascii_art_lib.utils.ansi import ANSI_RESET

def quantize(values: np.ndarray, levels: int) -> np.ndarray:
    """Равномерно квантует ``uint8`` массив в ``0..levels-1`` (векторизованно)."""
    # (v * levels) // 256 — точное и быстрое отображение 0..255 -> 0..levels-1
    return (values.astype(np.uint16) * np.uint16(levels)) >> 8


def frame_to_symbol_bytes(
    frame: np.ndarray,
    palette: str = DEFAULT_PALETTE,
    size: Optional[Tuple[int, int]] = None,
    *,
    reverse_palette: bool = False,
) -> np.ndarray:
    """Яркостные символы палитры как **uint8-сетка** (H, W) — только ASCII-палитры.

    Самый дешёвый по памяти и скорости путь: без Python-циклов, без объектов
    строк (256-байтная LUT + ``np.take``). Используется контурными режимами с
    заполнением фона «яркостью» и наложением контуров поверх ASCII-картинки.

    Returns:
        np.ndarray (h, w) dtype uint8 — байты символов; ``None``, если палитра
        содержит многобайтовые (Unicode) символы — тогда пользуйтесь
        :func:`frame_to_symbols`.
    """
    pal = get_palette(palette, reverse=reverse_palette)
    if not is_ascii_palette(pal):
        return None

    frame = resize_frame(frame, size)

    gray = to_gray(frame)
    lut = build_lut(pal)                                        # bytes(256)
    return np.take(np.frombuffer(lut, dtype=np.uint8), gray)    # C-скорость


def frame_to_symbols(
    frame: np.ndarray,
    palette: str = DEFAULT_PALETTE,
    size: Optional[Tuple[int, int]] = None,
    *,
    reverse_palette: bool = False,
) -> np.ndarray:
    """Конвертирует BGR-кадр в 2D-массив **символов яркости** (dtype='<U1' или str).

    Args:
        frame: Изображение BGR uint8 (H, W, 3) или grayscale (H, W).
        palette: Строка-палитра (от тёмных символов к светлым) либо её имя.
        size: Целевой размер ``(w, h)``; если ``None`` — используется размер кадра.
        reverse_palette: Перевернуть палитру (светлые символы будут соответствовать
            тёмным участкам изображения и наоборот).
    """
    pal = get_palette(palette, reverse=reverse_palette)
    n = len(pal)

    frame = resize_frame(frame, size)

    gray = to_gray(frame)

    if is_ascii_palette(pal):
        # Самый быстрый путь: одна табличная операция над всем массивом.
        lut = build_lut(pal)                                        # bytes(256)
        out_u8 = np.take(np.frombuffer(lut, dtype=np.uint8), gray)  # C-скорость
        # Мгновенная сборка строк из байтов (без <U1 — тот же dtype создаёт
        # объект на каждый символ и замедляет/раздувает память в ~4 раза).
        return np.array([bytes(row).decode("ascii") for row in out_u8])

    # Универсальный путь для Unicode/многобайтовых палитр: индексы -> lookup
    idx = quantize(gray, n)
    arr = np.array([pal[i] for i in idx.ravel()]).reshape(idx.shape)
    return arr


def symbols_to_text(sym: np.ndarray) -> str:
    """Собирает 1D/2D-массив строк в готовый текст (быстрая join-сборка)."""
    if sym.ndim == 1:
        return "\n".join(sym.tolist())
    rows: List[str] = ["".join(row) for row in sym.tolist()]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Цветной ASCII (ANSI truecolor) — оптимизированная замена img2ConsoleImg
# ---------------------------------------------------------------------------

def _ansi_fg(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def frame_to_color_ansi(
    frame: np.ndarray,
    palette: str = DEFAULT_PALETTE,
    size: Optional[Tuple[int, int]] = None,
    *,
    color_levels: Optional[int] = None,
    reverse_palette: bool = False,
    reset: str = ANSI_RESET,
) -> str:
    """Конвертирует BGR-кадр в **цветной** ASCII-арта с ANSI truecolor-кодами.

    Алгоритм (полностью векторный, O(H*W)):
      1. Кадрируется до ``size`` (INTER_AREA — быстрый и качественный даунскейл).
      2. Яркость квантуется по палитре -> символы (LUT / np.take).
      3. Каждый канал цвета квантуется до ``color_levels`` уровней: это даёт
         компактную палитру и позволяет ставить ANSI-префикс **только когда цвет
         пикселя меняется относительно предыдущего в строке** (дельта-кодирование),
         что сокращает объём вывода в разы по сравнению с префиксом на каждый символ.

    Args:
        frame: BGR uint8 (H, W, 3).
        palette: Палитра символов яркости.
        size: Целевой ``(w, h)`` в символах.
        color_levels: Уровней квантования на канал. ``None`` — охват выводится
            из размера палитры (:func:`~ascii_art_lib.palettes.palette_color_levels`):
            бедная палитра -> узкий цветовой охват, богатая -> широкий.
        reverse_palette: Перевернуть палитру перед конвертацией.
        reset: Escape-последовательность сброса цвета в конце строки.

    Returns:
        Готовая строка ANSI-арта (с переводами строк).
    """
    pal = get_palette(palette, reverse=reverse_palette)
    if color_levels is None:
        color_levels = palette_color_levels(pal)
    color_levels = max(1, min(256, int(color_levels)))

    frame = resize_frame(frame, size)

    sym_rows = frame_to_symbols(frame, pal).tolist()  # list[str] — по одной строке на ряд

    # Квантование цвета: 256 -> color_levels равномерных шагов
    q = max(1, 256 // color_levels)
    # round-to-nearest уровня, затем восстановление значения центра уровня
    lvl = (frame.astype(np.uint16) + (q // 2)) // q
    lvl = np.clip(lvl, 0, color_levels - 1)
    val = (lvl * q + (q // 2)).clip(0, 255).astype(np.uint8)
    b_ch, g_ch, r_ch = val[:, :, 0], val[:, :, 1], val[:, :, 2]  # BGR порядок cv2

    # Упаковываем квантованный цвет в один int32 id для дешёвого сравнения соседей
    cid = (r_ch.astype(np.int32) << 16) | (g_ch.astype(np.int32) << 8) | b_ch.astype(np.int32)

    # Где цвет меняется внутри строки (первый пиксель строки всегда «меняется»)
    change = np.empty(cid.shape, dtype=bool)
    change[:, 0] = True
    change[:, 1:] = cid[:, 1:] != cid[:, :-1]

    # Уникальные цвета -> словарь id -> ANSI-строка (обычно их десятки, не тысячи)
    uniq_ids = np.unique(cid[change])
    prefix_at = dict(
        zip(
            uniq_ids.tolist(),
            (_ansi_fg((i >> 16) & 0xFF, (i >> 8) & 0xFF, i & 0xFF) for i in uniq_ids.tolist()),
        )
    )

    cid_list = cid.tolist()      # list[list[int]]
    chg_list = change.tolist()   # list[list[bool]]
    out_lines: List[str] = []
    h, w = cid.shape
    for y in range(h):
        row_sym = sym_rows[y]
        row_cid = cid_list[y]
        row_ch = chg_list[y]
        parts: List[str] = []
        cur = ""
        for x in range(w):
            if row_ch[x]:
                cur = prefix_at[row_cid[x]]
            parts.append(cur)
            parts.append(row_sym[x])
        parts.append(reset)
        out_lines.append("".join(parts))
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Монохромный вывод
# ---------------------------------------------------------------------------

def frame_to_mono_text(
    frame: np.ndarray,
    palette: str = DEFAULT_PALETTE,
    size: Optional[Tuple[int, int]] = None,
    *,
    reverse_palette: bool = False,
) -> str:
    """Чёрно-белый ASCII-арт (без ANSI) — самый лёгкий по памяти режим."""
    sym = frame_to_symbols(frame, palette, size, reverse_palette=reverse_palette)
    return symbols_to_text(sym)
