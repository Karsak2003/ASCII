"""Палитра ориентации контуров: символ зависит от **наклона** линии, а не от яркости.

Когда включён режим контуров (:func:`ascii_art_lib.convert_image` с ``edges=True``),
результат ВСЕГДА рисуется собственной палитрой ориентации — яркостная палитра в
нём не участвует. Каждый пиксель контура получает символ, визуально сохраняющий
**направление** линии (как в классических ASCII-рисовальщиках, line-drawing /
slope-based palettes)::

    ~45°     0°      90°     135°        секторы касательной (шаг 22.5°, период 180°):
       \\    ___      |     //            0-1 -> '/'   2-3 -> '|'
        \\  /   \\    |    //             4-5 -> '-'   6-7 -> '\\'
         \\_/     \\__|___//              8-9 -> '_'   (горизонталь выпуклостью вниз)

Базовый набор (``mode="basic"``): ``/  -  _  |  \\``.

Расширенный набор (``mode="extended"``) добавляет символы **изгиба и углов** для
более сложных контуров. Дополнительно к направлению оценивается кривизна линии
(знак Laplacian'а яркости, нормированный на локальный контраст): прямые участки
остаются наклоном «-/_|\\», изогнутые получают форму, «раскрывающуюся» в сторону
вогнутости::

    вершина горба:        ^  v  <  >          (Up_Tee / DownTee / LeftTee / RightTee)
    чашка (впадина):      _  ‾                (LowLine / HighLine)
    боковые дуги:         (  )                ({  }   — широкие варианты)
    горизонтальные дуги:  [  ]                (левосторонние/правосторонние)
    стыки линий:          +  x                 (~горизонталь×~вертикаль, диагональ×диагональ)
    угловые стрелки:      ↑  ↓  ←  →           (резкий разворот контура)
    двойные линии:        ═  ║                 (сильный градиент: max/min по окрестности)

Символы вне ASCII (``‾ ↑ ↓ ← → ═ ║ ─``) хранятся в компактной кодовой сетке
``uint8`` парой escape+код и разворачиваются при сборке текста
(:func:`decode_grid`); там, где терминал/шрифт их не поддерживает, они выглядят
как прочерк или заменяются ближайшим ASCII-аналогом.

Чувствительность переключения на изгибные символы — параметр ``curve_threshold``
(меньше — больше изогнутых символов).

Два независимых аспекта вывода контуров:

* **Наложение на изображение** — отдельный флаг/API-параметр ``overlay``
  (по умолчанию **выключен**): контур накладывается ОТДЕЛЬНО поверх ASCII-версии
  оригинала — символы палитры ориентации замещают яркостные символы в позициях
  линий; цвет таких позиций — обычный ANSI-цвет кадра.
* **Заполнение фона** (актуально, когда наложение ВЫКЛЮЧЕНО) — ``fill``:
  ``"space"`` — пустой фон (по умолчанию), ``"."`` — точечная канва, либо любой
  пользовательский символ; спец-режим ``"brightness"`` — фон заполняется
  символами обычной яркостной палитры (контуры поверх ASCII-изображения).

Производительность/память: всё векторизовано (OpenCV/NumPy, O(H*W), C-скорость);
промежуточные float32-карты освобождаются внутри функции; результат — компактный
``uint8``-массив кодовой сетки (без <U1-объектов на символ).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .threshold_map import Gx as SOBEL_GX
from .threshold_map import Gy as SOBEL_GY

__all__ = [
    "EDGE_BASIC",
    "EDGE_EXTENDED",
    "EDGE_PALETTES",
    "EDGE_FILLS",
    "EDGE_FILL_PRESETS",
    "normalize_fill",
    "is_edge_fill_valid",
    "apply_edge_fill",
    "get_edge_palette",
    "edge_palette_symbols",
    "frame_to_edge_symbols",
    "edge_symbols_to_text",
    "frame_to_edge_ansi",
]

# ---------------------------------------------------------------------------
# Строковые представления палитр (справочно / для тестов)
# ---------------------------------------------------------------------------

#: Базовая палитра наклона: прямые сегменты под четырьмя углами
#: ('/' ~45°, '|' 90°, '-' 0°, '_' вогнутая чашка вниз, '\\' 135°).
EDGE_BASIC = "/|-_\\"

#: Расширенная палитра: наклон + символы изгиба/углов для сложных контуров.
#: Вершины горбов '^' 'v' '<' '>' и их двойные варианты 'V' 'A'; чашки '_' '‾';
#: дуги '(' ')' '[' ']' '{' '}'; стыки '+' 'x'; угловые стрелки '↑' '↓' '←' '→';
#: двойные линии при сильном контрасте '═' '║'.
EDGE_EXTENDED = "/|-_^vV<>()[]{}+xA‾↑↓←→═║"

EDGE_PALETTES: Dict[str, str] = {
    "basic": EDGE_BASIC,
    "extended": EDGE_EXTENDED,
}

#: Режимы заполнения, задаваемые одним словом (острые строки -- это сам символ
#: заполнения: "--edge-fill .", "--edge-fill '#'", "--edge-fill space").
EDGE_FILL_PRESETS = ("space", "brightness")

#: Допустимые режимы заполнения фона (когда контуры НЕ накладываются на оригинал).
#: ``"space"`` — пустой фон (по умолчанию), ``"brightness"`` — яркостная ASCII-канва,
#: либо любой одиночный символ-заполнитель ('.', '#', ':' …).
EDGE_FILLS = ("space", "brightness")


def normalize_fill(fill: Optional[str]) -> str:
    """Приводит значение заполнения фона к каноническому виду.

    Принимает: ``None``/``""``/``"space"``/``" "`` → ``" "`` (пустой фон);
    ``"brightness"`` → ``"brightness"``; любой непустой текст → его **первый
    символ** (``"."`` → ``"."``, ``"'.'\"`` → ``"."``). Пробелы по краям игнорируются.

    Returns:
        str длиной 1 (символ канвы) либо ``"brightness"``.
    """
    if fill is None:
        return " "
    f = str(fill).strip()
    if not f or f == "space":
        return " "
    if f == "brightness":
        return "brightness"
    if len(f) > 1:
        # Пользователь мог передать имя символа или кавычки — берём первый печатный символ
        printable = [c for c in f if c != " "]
        if not printable:
            return " "
        f = printable[0]
    return f


def is_edge_fill_valid(fill: str) -> bool:
    """True, если значение проходит валидацию CLI (--edge-fill)."""
    try:
        f = normalize_fill(fill)
    except Exception:  # noqa: BLE001
        return False
    if f in (" ", "brightness"):
        return True
    return len(f) == 1 and f != "\n" and f.isprintable()


def _fill_byte(fill: str) -> int:
    """Байт кодовой сетки для символа заполнения (ASCII-only, иначе '?')."""
    if fill == " ":
        return 0x20
    b = encode_char(fill)
    if len(b) == 1:
        return b[0]
    # Многобайтовый символ в uint8-сетке недоступен — заменяем на '?'
    return ord("?")


def apply_edge_fill(
    grid: np.ndarray,
    line_mask: np.ndarray,
    fill: Optional[str],
    *,
    unicode_grid: bool = False,
    brightness_bytes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Заменяет фон (пробелы) кодовой сетки контуров на символ заполнения.

    Применяется только когда контуры **не** накладываются на изображение
    (``overlay=False``): по умолчанию фон остаётся пустым (``"space"``).

    Args:
        grid: кодовая сетка из :func:`frame_to_edge_symbols` (H, W) или (H, 2W).
        line_mask: маска линий (H, W) bool — позиции, которые НЕ трогаем.
        fill: значение ``--edge-fill`` / ``fill=`` (см. :func:`normalize_fill`):
            ``"space"``/``None`` — без изменений; одиночный символ — однотонная
            канва; ``"brightness"`` — яркостные символы палитры (нужен
            ``brightness_bytes``).
        unicode_grid: ``True`` для extended-сетки (позиции символов выровнены
            парами байтов).
        brightness_bytes: uint8-сетка (H, W) яркостных ASCII-символов — источник
            для режима ``"brightness"`` (обычно из
            :func:`ascii_art_lib.converter.frame_to_symbol_bytes`). Если не
            передан и его невозможно получить — режим деградирует до пробелов.

    Returns:
        Новая кодовая сетка (или та же, если заполнение не требуется).
    """
    f = normalize_fill(fill)
    if f == " ":
        return grid
    bg = ~line_mask

    if f == "brightness":
        bb = brightness_bytes
        if bb is None or grid.shape[0] != bb.shape[0] or grid.shape[1] != bb.shape[1]:
            # Нет совместимой яркостной сетки — оставляем пустой фон
            return grid
        out = grid.copy()
        if unicode_grid:
            even = out[:, 0::2]
            odd = out[:, 1::2]
            ev_bb = bb[:, 0::2]
            od_bb = bb[:, 1::2]
            m_bg = bg[:, 0::2]
            even[m_bg] = ev_bb[m_bg]
            odd[m_bg] = od_bb[m_bg]
        else:
            out[bg] = bb[bg]
        return out

    byte = _fill_byte(f)
    out = grid.copy()
    if unicode_grid:
        even = out[:, 0::2]
        odd = out[:, 1::2]
        m_bg = bg[:, 0::2]
        even[m_bg] = byte
        odd[m_bg] = byte          # одиночный символ: дубль байта (выравнивание пар)
    else:
        out[bg] = byte
    return out

# ---------------------------------------------------------------------------
# Кодовая сетка uint8: ASCII-байт символа либо escape-пара 0x1B + код
# ---------------------------------------------------------------------------
# Компактный результат (H*W байт) не может хранить произвольный Unicode напрямую,
# поэтому не-ASCII символы кодируются парой ESC(0x1B)+код. Пробел (0x20) — фон.
_EXT_CODE: Dict[int, str] = {
    1: "\u203e",  # ‾ overline («чашка» горбом вверх)
    2: "\u2191",  # ↑ arrow up
    3: "\u2193",  # ↓ arrow down
    4: "\u2190",  # ← arrow left
    5: "\u2192",  # → arrow right
    6: "\u2550",  # ═ double horizontal
    7: "\u2551",  # ║ double vertical
    8: "\u2500",  # ─ light horizontal (запас)
    9: "\u2197",  # ↗ arrow north-east
    10: "\u2198",  # ↘ arrow south-east
    11: "\u2199",  # ↙ arrow south-west
    12: "\u2196",  # ↖ arrow north-west
}


def encode_char(ch: str) -> bytes:
    """Символ палитры -> байты кодовой сетки (одиночный байт или ESC+код)."""
    if ch == " ":
        raise ValueError("Пробел зарезервирован как фон и не кодируется в линиях")
    o = ord(ch)
    if 33 <= o <= 126 or 127 <= o <= 255:
        return bytes((o,))
    for code, ext in _EXT_CODE.items():
        if ext == ch:
            return b"\x1b" + bytes((code,))
    raise ValueError(f"Символ {ch!r} недоступен в компактной кодовой сетке контуров")


def decode_grid(arr: np.ndarray) -> list:
    """Кодовая сетка uint8 -> список строк текста (построчно, decode в Python лишь на escape)."""
    rows = []
    for row in arr.astype(np.uint8):
        m = row == 0x1B                       # позиции escape-префиксов
        if not m.any():
            rows.append(row.tobytes().decode("ascii"))
            continue
        codes = row.copy()
        codes[m] = 0x3F                       # временная заглушка под '?'
        s = codes.tobytes().decode("ascii")
        parts = []
        prev = 0
        for i in np.flatnonzero(m):
            parts.append(s[prev:i])
            parts.append(_EXT_CODE.get(int(row[i + 1]), "?"))
            prev = i + 2
        parts.append(s[prev:])
        rows.append("".join(parts))
    return rows


def grid_to_text(arr: np.ndarray) -> str:
    """Кодовая сетка uint8 -> готовый многострочный текст."""
    return "\n".join(decode_grid(arr))


def get_edge_palette(mode: str = "extended") -> str:
    """Возвращает строку-палитру ориентации контуров по имени режима."""
    try:
        return EDGE_PALETTES[mode]
    except KeyError:
        raise ValueError(
            f"Неизвестный режим палитры контуров: {mode!r} (ожидается {sorted(EDGE_PALETTES)})"
        ) from None


def edge_palette_symbols(mode: str = "extended") -> Tuple[str, ...]:
    """Символы палитры ориентации кортежем."""
    return tuple(get_edge_palette(mode))


# ---------------------------------------------------------------------------
# Внутренние утилиты
# ---------------------------------------------------------------------------

def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _odd(v: int) -> int:
    v = int(v)
    return v if v % 2 == 1 else v + 1


def _hysteresis_mask(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Морфологическая реконструкция: слабые пиксели, связанные с сильными."""
    seed = strong.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    max_iters = min(64, max(seed.shape) // 2 + 1)
    for _ in range(max_iters):
        dilated = cv2.dilate(seed, kernel, iterations=1)
        candidate = np.where(weak & (dilated > 0), 255, 0).astype(np.uint8)
        if np.array_equal(candidate, seed):
            break
        seed = candidate
    return seed > 0


# ---------------------------------------------------------------------------
# Кодовые таблицы символов (значения — байты кодовой сетки, см. encode_char)
# ---------------------------------------------------------------------------

def _c(ch: str) -> bytes:
    """Символ палитры -> байты кодовой сетки (одиночный байт либо ESC+код)."""
    return encode_char(ch)


def _row(*chars: str) -> bytes:
    """Строка таблицы из 4 символов (col 0..3) в байты кодовой сетки."""
    assert len(chars) == 4
    return b"".join(encode_char(c) for c in chars)


# Базовый набор наклона: сектор касательной (шаг 22.5°, период 180°) -> символ.
#   0:[0,22.5)   '/'      4:[90,112.5)   '|'
#   1:[22.5,45)  '/'      5:[112.5,135)  '\'
#   2:[45,67.5)  '-'      6:[135,157.5)  '\'
#   3:[67.5,90)  '-'      7:[157.5,180)  '_'  (горизонталь выпуклостью вниз)
_SECTOR_CH = ("/", "/", "-", "-", "|", "\\", "\\", "_")
SECTOR_CHARS: Tuple[str, ...] = _SECTOR_CH
#: Байты базовых символов по секторам (для escape-пар — первый байт пары).
_SECTOR_BYTE = np.array([int(encode_char(c)[-1]) for c in _SECTOR_CH], dtype=np.uint8)

#: Группа линии по сектору касательной: 0:'/' 1:'-'/'_' 2:'|' 3:'\'
GROUP_OF_SECTOR = np.array([0, 0, 1, 1, 2, 3, 3, 1], dtype=np.int32)

# --- Расширенная палитра: символы изгиба и наклона ---------------------------
# Направление горба (выпуклости) — октант нормали к яркому фону:
#   0 ↑вверх  1 ↗СЕ  2 →вправо  3 ↘ЮВ  4 ↓вниз  5 ↙ЮЗ  6 ←влево  7 ↖СЗ
# Для впадины (curv < 0) берётся противоположный октант (+4).
# col таблицы изгибов = группа линии: 0:'/' 1:'-'/'_' 2:'|' 3:'\'
#
#   ^ v < >        вершины горбов (прямые направления)
#   A V            «двойная» вершина при очень сильном изломе
#   _ ‾            чашки (впадины горизонталей): нижняя / верхняя
#   ( ) [ ] { }    боковые дуги диагональных и вертикальных участков
#   ↑ ↓ ← →        угловые стрелки — резкие развороты контура
#   + x            стыки линий (разные группы в одной позиции)
_CURVE_ROWS_CH = (
    ("(", "[", "{", "("),   # 0 горб вверх:      '/'->'(', '-'->'[', '|'->'{', '\'->'('
    ("/", "(", "{", "]"),   # 1 горб СЕ:         наклон сохраняется, дуги по сторонам
    (")", "‾", "}", ")"),   # 2 горб вправо:     '/'->')', '-'->'‾', '|'->'}', '\'->')'
    ("[", ")", "]", "\\"),  # 3 горб ЮВ
    (")", "]", "}", ")"),   # 4 горб вниз:       '/'->')', '-'->']', '|'->'}', '\'->')'
    ("\\", "(", "]", "/"),  # 5 горб ЮЗ
    ("(", "{", "[", "("),   # 6 горб влево
    ("/", ")", "}", "]"),   # 7 горб СЗ
)
#: Строки таблицы изгибов как текст (справочно/тесты).
CURVE_TABLE_CHARS: Tuple[Tuple[str, ...], ...] = tuple(tuple(r) for r in _CURVE_ROWS_CH)


def _build_curve_table() -> np.ndarray:
    """Таблица «октант горба x группа линии» -> байты кодовой сетки.

    Формат строки (8 байт): байты 0..3 — первые байты символов col 0..3
    (для ASCII — сам символ, для unicode — ESC); байты 4..7 — вторые
    байты (unicode-код) либо дубль ASCII-байта. Так позиция символа
    в строке всегда кратна 2 и decode_grid корректно разворачивает пары.
    """
    t = np.zeros((8, 8), dtype=np.uint8)
    for i, row in enumerate(_CURVE_ROWS_CH):
        for j, ch in enumerate(row):
            b = encode_char(ch)
            if len(b) == 1:
                t[i, j] = b[0]
                t[i, 4 + j] = b[0]
            else:
                t[i, j] = b[0]          # ESC
                t[i, 4 + j] = b[1]      # unicode-код
    return t


_CURVE_TABLE = _build_curve_table()

#: Угловые стрелки по октанту направления горба (резкие развороты контура).
ARROW_CHARS: Tuple[str, ...] = ("↑", "↗", "→", "↘", "↓", "↙", "←", "↖")
_ARROW_BYTES = np.frombuffer(b"".join(encode_char(c) for c in ARROW_CHARS),
                             dtype=np.uint8).reshape(8, 2)   # все — escape-пары

#: Порог «сильного излома» (относительная кривизна): выше — ставим стрелку.
SHARP_CORNER_FACTOR = 6.0

#: Стыки линий (пересечения разных групп в 3×3 соседстве): '+' или 'x'.
JOIN_PLUS_CHAR, JOIN_X_CHAR = "+", "x"
JOIN_PLUS_CODE = ord(JOIN_PLUS_CHAR)
JOIN_X_CODE = ord(JOIN_X_CHAR)

#: Двойные линии при очень сильном контрасте (участок входит в «сильные»):
#: одиночные '-'/'_'/'|' заменяются на '═'/'║'.
DOUBLE_LINE_CHARS: Tuple[str, ...] = ("═", "║")
_DOUBLE_H = ord("-"); _DOUBLE_U = ord("_"); _DOUBLE_V = ord("|")
_DLINE_H_CODE = int(encode_char("═")[-1]); _DLINE_V_CODE = int(encode_char("║")[-1])


# ---------------------------------------------------------------------------
# Ядро: кадр -> кодовая сетка символов ориентации + вспомогательные карты
# ---------------------------------------------------------------------------
def _edge_maps(
    frame: np.ndarray,
    size: Optional[Tuple[int, int]],
    *,
    mode: str,
    low_threshold: int,
    high_threshold: int,
    blur_ksize: int,
    curve_threshold: float,
    method: str,
    sharp_factor: float = SHARP_CORNER_FACTOR,
    joins: bool = True,
    double_lines: bool = True,
) -> dict:
    """Единый расчёт геометрии контура (используется всеми режимами вывода).

    Args:
        sharp_factor: Во сколько раз ``curve_threshold`` нужно превысить, чтобы
            участок считался резким изломом и получил угловую стрелку
            ('\u2191 \u2193 \u2190 \u2192 \u2197 \u2198 \u2199 \u2196').
        joins: Помечать ли стыки двух групп линий символами '+' / 'x'.
        double_lines: Заменять ли одиночные '-'/'_'/'|' на двойные '\u2550'/'\u2551'
            там, где градиент локально максимален и входит в «сильные» контуры.

    Returns:
        dict с полями: ``mask`` (bool HxW), ``sector``/``group``/``bump`` (int32),
        ``grid`` — кодовая сетка uint8 символов линий (фон = пробел; escape-пары
        выровнены по чётным позициям), ``line_mask`` — маска непустых позиций,
        ``gray`` (float32 после размытия), ``mag_u8``, ``strong``.
    """
    if mode not in EDGE_PALETTES:
        raise ValueError(
            f"Неизвестный режим палитры контуров: {mode!r} (ожидается {sorted(EDGE_PALETTES)})"
        )

    if size is not None and (frame.shape[1], frame.shape[0]) != tuple(size):
        frame = cv2.resize(frame, tuple(int(v) for v in size), interpolation=cv2.INTER_AREA)

    gray = _to_gray(frame)
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)

    # 1. Подавление шума (влияет и на оценку кривизны).
    if blur_ksize and blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (_odd(blur_ksize), _odd(blur_ksize)), 0)

    # 2. Градиенты Sobel — те же ядра, что и в ascii_art_lib.edges.
    gx = cv2.filter2D(gray, cv2.CV_32F, SOBEL_GX)
    gy = cv2.filter2D(gray, cv2.CV_32F, SOBEL_GY)
    mag = np.empty_like(gx)
    cv2.magnitude(gx, gy, mag)

    # 3. Нормализация величины градиента к 0..255.
    mmax = float(mag.max()) if mag.size else 0.0
    if mmax > 255.0:
        mag *= 255.0 / mmax
    mag_u8 = mag.astype(np.uint8)

    # 4. Двойной порог (+ гистерезис) -> маска контура.
    strong = mag_u8 >= int(high_threshold)
    weak = mag_u8 >= int(low_threshold)
    if method == "sobel" or not strong.any():
        mask = weak
    elif not weak.any():
        mask = strong
    else:
        mask = _hysteresis_mask(strong, weak)

    # 5. Угол КАСАТЕЛЬНОЙ (период pi) -> сектор/группа; угол НОРМАЛИ (период 2pi)
    #    -> октант направления роста яркости.
    angle = np.arctan2(-gx, gy) % math.pi
    nrm = np.arctan2(gy, gx) % (2.0 * math.pi)     # 0 = вправо, против часовой
    nsec = np.clip((nrm * (4.0 / math.pi)).astype(np.int32), 0, 7)
    sector = np.clip((angle * (8.0 / math.pi)).astype(np.int32), 0, 7)
    group = GROUP_OF_SECTOR[sector]

    # 6. Базовый наклон (табличная операция, без Python-цикла по пикселям).
    base_byte = _SECTOR_BYTE[sector]
    out = base_byte.copy()

    # 7. Кривизна: Laplacian яркости, нормированный на локальный контраст.
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    curv = -lap                                     # >0 — горб выпуклостью к свету
    rel = np.abs(curv) / (mag + 1e-3)
    ct = float(curve_threshold)
    is_curve = mask & (rel > ct)
    is_sharp = mask & (rel > ct * max(1.0, float(sharp_factor)))
    bump = np.where(curv >= 0, nsec, (nsec + 4) & 7)

    second = np.zeros(out.shape, dtype=np.uint8)   # второй байт escape-пары

    if mode == "extended":
        # 7a. Изгибы -> дуги/вершины/чашки из таблицы (col = группа линии).
        if is_curve.any():
            rows = _CURVE_TABLE[bump]               # (H, W, 8): байты 0..3 + 4..7
            cols = np.arange(rows.shape[2])         # 0..7
            sel = ((cols >= group[..., None]) & (cols < group[..., None] + 1)) | \
                  ((cols >= 4 + group[..., None]) & (cols < 5 + group[..., None]))
            pair = rows[sel].reshape(rows.shape[:2] + (2,))
            hi, lo = pair[..., 0], pair[..., 1]     # ASCII: hi==lo; unicode: ESC+код
            out = np.where(is_curve, hi, out)
            second = np.where(is_curve, lo, second)

            # 7b. Резкие изломы -> угловые стрелки (всегда escape-пары ESC+код).
            if is_sharp.any():
                a_pair = _ARROW_BYTES[bump]         # (H, W, 2)
                out = np.where(is_sharp, a_pair[..., 0], out)
                second = np.where(is_sharp, a_pair[..., 1], second)

        # 7c0. Прямые участки (не изгибы): второй байт — дубль одиночного символа,
        #      чтобы позиции escape-пар оставались выровнены по чётным байтам.
        second = np.where(out == 0x1B, second, out)

        # 7c. Стыки линий: разные группы базового наклона в 3x3 -> '+' или 'x'.
        if joins:
            g = np.pad(group, 1, mode="constant", constant_values=-1)
            nb_groups = np.stack([g[:-2, :-2], g[1:-1, :-2], g[2:, :-2],
                                  g[:-2, 1:-1], g[2:, 1:-1],
                                  g[:-2, 2:], g[1:-1, 2:], g[2:, 2:]], axis=0)
            same = (nb_groups == group[None]) | (nb_groups < 0)
            has_diff = (~same).any(axis=0) & mask
            slash_nb = ((nb_groups == 0) & ~same).any(axis=0)
            back_nb = ((nb_groups == 3) & ~same).any(axis=0)
            horiz_nb = ((nb_groups == 1) & ~same).any(axis=0)
            vert_nb = ((nb_groups == 2) & ~same).any(axis=0)
            x_join = has_diff & slash_nb & back_nb
            p_join = has_diff & (horiz_nb | vert_nb) & ~x_join
            join_sel = x_join | p_join
            join_byte = np.where(x_join, JOIN_X_CODE, JOIN_PLUS_CODE)
            out = np.where(join_sel, join_byte, out)
            second = np.where(join_sel, join_byte, second)   # одиночные байты: дубль

        # 7d. Двойные линии при сильном контрасте: '-'/'_' -> '\u2550', '|' -> '\u2551'.
        if double_lines:
            mx = cv2.dilate(mag_u8, np.ones((3, 3), np.uint8))
            local_peak = (mag_u8 >= mx) & strong & mask
            h_line = local_peak & ((out == _DOUBLE_H) | (out == _DOUBLE_U))
            v_line = local_peak & (out == _DOUBLE_V)
            if h_line.any() or v_line.any():
                out = np.where(h_line, 0x1B, out)
                second = np.where(h_line, _DLINE_H_CODE, second)
                out = np.where(v_line, 0x1B, out)
                second = np.where(v_line, _DLINE_V_CODE, second)

        # 8. Сборка кодовой сетки: escape-пары занимают 2 байта строки, поэтому
        #    каждая позиция символа кодируется парой (hi, lo) на чётных смещениях.
        grid = np.full((out.shape[0], out.shape[1] * 2), 0x20, dtype=np.uint8)
        esc = out == 0x1B
        even = grid[:, 0::2]; odd = grid[:, 1::2]
        even[:] = np.where(mask, out, 0x20)
        odd[:] = np.where(mask & esc, second, 0x20)
        line_mask = mask
    else:
        grid = np.where(mask, out, np.uint8(0x20)).astype(np.uint8)
        line_mask = mask

    return dict(
        mask=mask, sector=sector, group=group, bump=bump, is_curve=is_curve,
        grid=grid, line_mask=line_mask, gray=gray, mag_u8=mag_u8, strong=strong,
        frame=frame, unicode_grid=(mode == "extended"),
    )


def frame_to_edge_symbols(
    frame: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
    *,
    mode: str = "extended",
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_ksize: int = 5,
    curve_threshold: float = 0.5,
    method: str = "canny",
    sharp_factor: float = SHARP_CORNER_FACTOR,
    joins: bool = True,
    double_lines: bool = True,
) -> np.ndarray:
    """Конвертирует кадр в компактную **кодовую сетку uint8** символов ориентации.

    Фон — пробелы (0x20); позиции контура — символ, чей наклон соответствует
    локальному направлению линии, а форма — её кривизне (режим ``extended``).
    В ``extended`` не-ASCII символы хранятся парами ESC(0x1B)+код (позиции
    символов выровнены по чётным байтам); текст собирается функциями
    :func:`decode_grid` / :func:`edge_symbols_to_text`.

    Args:
        frame: BGR (H, W, 3) или grayscale (H, W) uint8.
        size: Целевой ``(w, h)``; ``None`` — размер кадра как есть.
        mode: ``"basic"`` — только ``/ - _ | \\``; ``\"extended\"`` — добавляются
            вершины ``^ v < >``, чашки ``_ \u203e``, дуги ``( ) [ ] { }``,
            стыки ``+ x``, угловые стрелки ``\u2191 \u2193 \u2190 \u2192 \u2197 \u2198 \u2199 \u2196`` и двойные
            линии ``\u2550 \u2551`` для контрастных участков.
        low_threshold / high_threshold: Пороги двойной фильтрации (как в Canny).
        blur_ksize: Гауссово размытие перед детекцией (0 — выключить).
        curve_threshold: Чувствительность определения изгиба (рекомендуется
            0.3..1.5). Меньше значение — больше изогнутых символов.
        method: ``"canny"`` (с гистерезисом) или ``"sobel"`` (один порог).
        sharp_factor: Множитель порога «резкого излома» для угловых стрелок.
        joins: Ставить ли символы стыков ``+`` / ``x``.
        double_lines: Ставить ли двойные линии ``\u2550 \u2551`` на пиках контраста.

    Returns:
        np.ndarray (h, w[, 2w]) dtype uint8 — кодовая сетка (:func:`decode_grid`).
    """
    return _edge_maps(
        frame, size,
        mode=mode, low_threshold=low_threshold, high_threshold=high_threshold,
        blur_ksize=blur_ksize, curve_threshold=curve_threshold, method=method,
        sharp_factor=sharp_factor, joins=joins, double_lines=double_lines,
    )["grid"]


def edge_symbols_to_text(sym: np.ndarray) -> str:
    """Кодовая сетка uint8 -> готовый текст (поддержка escape-пар Unicode)."""
    return "\n".join(decode_grid(sym))


def expand_edge_grid(grid: np.ndarray, width: int) -> np.ndarray:
    """Разворачивает компактную кодовую сетку контуров в полноширинную.

    Расширенная палитра хранит не-ASCII символы парами ESC+код, поэтому её
    сетка имеет ширину ``2*w``; при прямом выводе текста такие строки вдвое
    шире яркостного ASCII и **искажают пропорции** картинки. Эта функция
    приводит любую сетку к форме ``(h, width)`` — по одному текстовому
    символу на ячейку изображения:

    * escape-пара ``(ESC, code)`` сворачивается в один Unicode-символ
      (его байт-представление занимает чётные позиции сетки);
    * одиночные ASCII-байты дублируются (занимают обе позиции пары);
    * если сетка уже полноширинная — возвращаются первые ``width`` столбцов;
    * иначе результат дополняется пробелами / обрезается до ``width``.

    Args:
        grid: кодовая сетка ``(h, w)`` или ``(h, 2w)`` dtype uint8.
        width: целевая ширина вывода в символах (обычно ``w`` из ``size=(w, h)``).

    Returns:
        np.ndarray ``(h, width)`` dtype uint8 с одиночными байтами символов
        (Unicode развёрнут в UTF-8), готовый к :func:`decode_grid` / печати.
    """
    g = np.asarray(grid, dtype=np.uint8)
    if g.ndim != 2:
        raise ValueError("Ожидается 2D кодовая сетка")
    h = g.shape[0]
    out = np.full((h, max(int(width), 0)), 0x20, dtype=np.uint8)
    if g.size == 0 or out.size == 0:
        return out

    # Разворачиваем сетку построчно в список строк (escape-пары -> 1 символ)
    rows = decode_grid(g)
    for y, row in enumerate(rows):
        s = row[: width]
        if len(s):
            b = s.encode("utf-8", "replace")
            out[y, : len(b)] = np.frombuffer(b, dtype=np.uint8)[: width]
    return out


# ---------------------------------------------------------------------------
# Цветной ANSI-вариант (дельта-кодирование цвета, как в converter.frame_to_color_ansi)
# ---------------------------------------------------------------------------

def frame_to_edge_ansi(
    frame: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
    *,
    mode: str = "extended",
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_ksize: int = 5,
    curve_threshold: float = 0.5,
    method: str = "canny",
    color_levels: int = 32,
    bg_mode: str = "space",
    reset: str = "\033[0m",
) -> str:
    """Контурная ASCII-графика с **цветом оригинала**: символ — наклон линии,
    ANSI-цвет — квантованный цвет исходного кадра в этой точке.

    Args:
        frame: BGR uint8 (H, W, 3).
        size: Целевой ``(w, h)`` в символах.
        mode / low_threshold / high_threshold / blur_ksize / curve_threshold /
            method: см. :func:`frame_to_edge_symbols`.
        color_levels: Уровней квантования на канал. В высокоуровневом API
            (``convert_image(edges=True)``) охват по умолчанию выводится из
            размера самой палитры ориентации (``extended`` → 14 уникальных
            символов ≈ 9 уровней/канал, ``basic`` → 4 ≈ 6); здесь — прямой
            параметр с тем же правилом вывода из палитры.
        bg_mode: ``"space"`` — фон пропускается целиком (компактный вывод);
            ``"dark"`` — фон рисуется тёмным цветом оригинала и пробелами
            (фон читается как силуэт).
        reset: Escape-последовательность сброса цвета в конце строки.

    Returns:
        Готовая многострочная ANSI-строка.
    """
    import cv2  # noqa: F401  (уже импортирован на уровне модуля)

    if size is not None and (frame.shape[1], frame.shape[0]) != tuple(size):
        frame = cv2.resize(frame, tuple(int(v) for v in size), interpolation=cv2.INTER_AREA)

    sym = frame_to_edge_symbols(
        frame, None,
        mode=mode, low_threshold=low_threshold, high_threshold=high_threshold,
        blur_ksize=blur_ksize, curve_threshold=curve_threshold, method=method,
    )

    # Квантованный цвет оригинала -> id для дельта-кодирования
    levels = max(1, min(256, int(color_levels)))
    q = max(1, 256 // levels)
    lvl = (frame.astype(np.uint16) + (q // 2)) // q
    lvl = np.clip(lvl, 0, levels - 1)
    val = (lvl * q + (q // 2)).clip(0, 255).astype(np.uint8)
    b_ch, g_ch, r_ch = val[:, :, 0], val[:, :, 1], val[:, :, 2]
    cid = (r_ch.astype(np.int32) << 16) | (g_ch.astype(np.int32) << 8) | b_ch.astype(np.int32)

    edge = sym != 0x20
    if bg_mode == "dark":
        # Тёмный вариант цвета фона: чтобы пробелы-фон создавали «глубину».
        dark = (val.astype(np.uint16) >> 2).astype(np.uint8)
        bd, gd, rd = dark[:, :, 0], dark[:, :, 1], dark[:, :, 2]
        did = (rd.astype(np.int32) << 16) | (gd.astype(np.int32) << 8) | bd.astype(np.int32)
        cid = np.where(edge, cid, did)
        emit = np.ones(sym.shape, dtype=bool)
    elif bg_mode == "space":
        emit = edge
    else:
        raise ValueError(f"Неизвестный bg_mode: {bg_mode!r} ('space'/'dark')")

    # Где ставить ANSI-префикс: первый эмитированный символ строки и смены цвета
    change = np.empty(cid.shape, dtype=bool)
    change[:, 0] = True
    change[:, 1:] = cid[:, 1:] != cid[:, :-1]
    change &= emit

    uniq_ids = np.unique(cid[change])
    prefix_at = dict(
        zip(
            uniq_ids.tolist(),
            (f"\033[38;2;{(i >> 16) & 0xFF};{(i >> 8) & 0xFF};{i & 0xFF}m" for i in uniq_ids.tolist()),
        )
    )

    sym_list = sym.tolist()      # list[list[int]] байтов
    cid_list = cid.tolist()
    chg_list = change.tolist()
    emit_list = emit.tolist()
    out_lines = []
    h, w = sym.shape
    for y in range(h):
        row_sym, row_cid = sym_list[y], cid_list[y]
        row_ch, row_em = chg_list[y], emit_list[y]
        parts = []
        run = ""                  # накопитель идущих подряд символов одного цвета
        cur = ""                  # ANSI-префикс текущего цвета
        prev_emitted = False      # был ли предыдущий символ эмитирован (не фон)
        for x in range(w):
            if not row_em[x]:
                # Фон: в compact-режиме пропускаем ЦЕЛИКОМ — но если после фона
                # снова идёт линия, обязательно добавляем ровно один пробел-
                # разделитель, иначе символы «склеиваются» и геометрия теряется.
                if prev_emitted and bg_mode == "space":
                    if run:
                        parts.append(run)
                        run = ""
                    parts.append(" ")
                prev_emitted = False
                continue
            if row_ch[x] or not prev_emitted:
                # Новый цвет (или старт линии после фона) — закрываем текущий ран
                if run:
                    parts.append(run)
                run = ""
                cur = prefix_at[row_cid[x]]
            run += chr(row_sym[x])
            prev_emitted = True
        if run:
            parts.append(run)
        parts.append(reset)
        out_lines.append("".join(parts))
    return "\n".join(out_lines)
