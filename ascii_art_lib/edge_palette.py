"""Палитра ориентации контуров: символ зависит от **наклона** линии, а не от яркости.

Обычный режим ``edges + edge_mode="lines"`` раскрашивает контуры символами
*яркости* (белая линия -> самый светлый символ палитры). Этот модуль реализует
альтернативный подход: каждый пиксель контура получает символ, визуально
сохраняющий **направление** линии — как в классических ASCII-рисовальщиках
(line-drawing / slope-based palettes)::

        ~45°    0°/180°   135°          секторы угла касательной:
           \\     |    /                  0-1  ->  '/'
            \\    |   /                   2-3  ->  '|'
             \\   |  /                    4-5  ->  '-'
              .......                    6-7  ->  '\\'

Базовый набор (``mode="basic"``): ``/ - \\ |``.

Расширенный набор (``mode="extended"``) добавляет парные скобки для *изогнутых*
участков контура: дополнительно оценивается кривизна линии (знак Laplacian'а
яркости, нормированный на локальный контраст). Прямые сегменты остаются
«-/|\\\\», а изгибы получают символ, «раскрывающийся» в сторону вогнутости::

    горб вверх '^'  , горб вниз 'v', горб влево '<', горб вправо '>'
    диагональные изгибы '(' ')' '[' ']' '{' '}'

Чувствительность переключения на скобки — параметр ``curve_threshold``
(меньше — больше изогнутых символов).

Производительность/память: всё векторизовано (OpenCV/NumPy, O(H*W), C-скорость);
промежуточные float32-карты освобождаются внутри функции; результат — компактный
``uint8``-массив ASCII-байтов (без <U1-объектов на символ).
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
    "get_edge_palette",
    "edge_palette_symbols",
    "frame_to_edge_symbols",
    "edge_symbols_to_text",
    "frame_to_edge_ansi",
]

# ---------------------------------------------------------------------------
# Строковые представления палитр (справочно / для тестов)
# ---------------------------------------------------------------------------

#: Базовая палитра наклона: прямые сегменты под четырьмя углами.
EDGE_BASIC = "/|-\\"

#: Расширенная палитра: прямые + парные скобки для изогнутых участков.
EDGE_EXTENDED = "/|-\\^v<>()[]{}"

EDGE_PALETTES: Dict[str, str] = {
    "basic": EDGE_BASIC,
    "extended": EDGE_EXTENDED,
}


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


# Таблица «сектор касательной (0..7) -> байт базового символа».
# Угол считывается с шагом 22.5°, период 180° (линия не имеет «стрелки»):
#   0:[0,22.5)  '/'      4:[90,112.5)  '-'
#   1:[22.5,45) '/'      5:[112.5,135) '-'
#   2:[45,67.5) '|'      6:[135,157.5) '\\'
#   3:[67.5,90) '|'      7:[157.5,180) '\\'
_SECTOR_BYTE = np.array(
    [ord("/"), ord("/"), ord("|"), ord("|"), ord("-"), ord("-"), ord("\\"), ord("\\")],
    dtype=np.uint8,
)

# Таблица скобочных символов для изогнутых участков: [куда смотрит горб][группа линии].
# группа: 0:'/' 1:'|' 2:'-' 3:'\\'. Строки таблицы соответствуют октанту нормали
# к яркому фону (горб указывает в эту сторону), col 0/3 — диагональные изгибы,
# col 1 — вертикальные дуги '(' ')', col 2 — горизонтальные дуги '[' ']' '{' '}'.
# _CURVE_ROWS[i] — i-й октант направления выпуклости (0 = вверх, далее по часовой).
_CURVE_ROWS = (
    b"^([)",   # горб вверх
    b"/({[",   # северо-восток
    b">)/[",   # вправо
    b"\\})/",  # юго-восток
    b"v}])",   # вниз
    b"|](\\" ,  # юго-запад
    b"<)[\\",  # влево
    b"[/{\\",   # северо-запад
)
_CURVE_TABLE = np.frombuffer(b"".join(_CURVE_ROWS), dtype=np.uint8).reshape(8, 4)


# ---------------------------------------------------------------------------
# Ядро: кадр -> массив байтов-символов ориентации
# ---------------------------------------------------------------------------

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
) -> np.ndarray:
    """Конвертирует кадр в 2D-массив uint8 ASCII-байтов по **ориентации контуров**.

    Фон — пробелы (0x20); пиксели контура — символ, чей наклон соответствует
    локальному направлению линии (и форма — её кривизне в режиме ``extended``).

    Args:
        frame: BGR (H, W, 3) или grayscale (H, W) uint8.
        size: Целевой ``(w, h)``; ``None`` — размер кадра как есть.
        mode: ``"basic"`` — только ``/ - \\ |``; ``"extended"`` — добавляются
            ``^ v < > ( ) [ ] { }`` для изогнутых участков.
        low_threshold / high_threshold: Пороги двойной фильтрации (как в Canny).
        blur_ksize: Гауссово размытие перед детекцией (0 — выключить).
        curve_threshold: Чувствительность определения изгиба (рекомендуется
            0.3..1.5). Меньше значение — больше участков получает скобки.
        method: ``"canny"`` (с гистерезисом) или ``"sobel"`` (один порог).

    Returns:
        np.ndarray (h, w) dtype uint8 — ASCII-байты (можно декодировать построчно).
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

    # 5. Угол КАСАТЕЛЬНОЙ линии (перпендикуляр градиенту), период pi;
    #    и угол НОРМАЛИ (направление роста яркости), период 2pi -> октант 0..7.
    angle = np.arctan2(-gx, gy)
    angle %= math.pi
    nrm = np.arctan2(gy, gx) % (2.0 * math.pi)          # 0 = вправо, против часовой
    nsec = np.clip((nrm * (4.0 / math.pi)).astype(np.int32), 0, 7)

    # 6. Сектор (0..7) -> базовый символ наклона (табличная операция, без циклов).
    sector = np.clip((angle * (8.0 / math.pi)).astype(np.int32), 0, 7)
    out = _SECTOR_BYTE[sector]

    # 7. Extended: изогнутые участки перекрашиваем скобочными символами.
    if mode == "extended":
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        curv = -lap                                   # >0 — горб выпуклостью к свету
        mag_f = mag.astype(np.float32) + 1e-3         # нормализованная величина (до u8!)
        rel = np.abs(curv) / mag_f
        is_curve = mask & (rel > float(curve_threshold))
        if is_curve.any():
            line_group = sector >> 1                  # 0:'/' 1:'|' 2:'-' 3:'\\'
            # Куда смотрит горб: по нормали (curv>0) или против неё (curv<0).
            bump = np.where(curv >= 0, nsec, (nsec + 4) & 7)
            chosen = _CURVE_TABLE[bump, line_group]
            out = np.where(is_curve, chosen, out)

    # 8. Фон -> пробелы.
    return np.where(mask, out, np.uint8(0x20)).astype(np.uint8)


def edge_symbols_to_text(sym: np.ndarray) -> str:
    """Собирает uint8-массив байтов-символов в готовый текст (построчный decode)."""
    rows = [bytes(row).decode("ascii") for row in sym]
    return "\n".join(rows)


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
        color_levels: Уровней квантования на канал (для контурного режима
            зависимость «охват <- палитра яркости» неприменима, поэтому здесь
            фиксированный разумный дефолт 32; можно переопределить).
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
        cur = ""
        for x in range(w):
            if not row_em[x]:
                continue  # фон пропускаем (compact) — линии «плавают» на пустоте
            if row_ch[x]:
                cur = prefix_at[row_cid[x]]
            parts.append(cur)
            parts.append(chr(row_sym[x]))
        parts.append(reset)
        out_lines.append("".join(parts))
    return "\n".join(out_lines)
