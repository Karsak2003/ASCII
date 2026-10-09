"""Абстракции вывода: подсветка синтаксиса (ANSI-цвет) и интерфейс рендера.

Модуль вводит два уровня абстракции, вокруг которых собраны конкретные
реализации (``core.converter``, ``edges.palette``, ``rendering.console``):

* :class:`SyntaxHighlighter` — абстрактный «подсветчик» ASCII-арта: превращает
  кодовую сетку символов + цвет оригинала в ANSI-текст. Конкретные стратегии:

  - :class:`TruecolorHighlighter` — RGB truecolor с дельта-кодированием
    (префикс ставится только при смене цвета);
  - :class:`NoHighlighter` — без цветовых кодов (монохром).

* :class:`BaseRenderer` — абстрактный вывод готового текста (консоль/файл/NUL);
  наследник :class:`~ascii_art_lib.rendering.console.ConsoleRenderer` печатает
  в терминал, :class:`NullRenderer` нужен для тестов и «тихих» прогонов CLI.

Все сигнатуры аннотированы типами; классы используют ``abc.ABCMeta`` и
``@abstractmethod`` — собственные расширения вывод/подсветки достаточно
унаследовать от этих баз.
"""

from __future__ import annotations

import abc
from typing import Iterable, List, Optional, Sequence

import numpy as np

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
]

#: Escape-последовательность сброса всех атрибутов ANSI.
ANSI_RESET: str = "\033[0m"


# ---------------------------------------------------------------------------
# Низкоуровневые ANSI-примитивы (общие для всех реализаций подсветки)
# ---------------------------------------------------------------------------

def ansi_fg_truecolor(r: int, g: int, b: int) -> str:
    """SGR-префикс foreground truecolor ``\\033[38;2;R;G;Bm``."""
    return f"\033[38;2;{int(r)};{int(g)};{int(b)}m"


def strip_ansi(text: str) -> str:
    """Убирает все ANSI escape-последовательности из строки (для тестов/замеров)."""
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\x1b":
            j = i + 1
            while j < n and not ("@" <= text[j] <= "~"):
                j += 1
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def quantize_color(frame_rgb: np.ndarray, levels: int) -> np.ndarray:
    """Равномерно квантует каждый канал кадра ``(H, W, 3)`` до ``levels`` уровней.

    Возвращает uint8-массив тех же размеров со значениями в центрах уровней —
    именно они становятся палитрой ANSI-цветов вывода.
    """
    levels = max(1, min(256, int(levels)))
    q = max(1, 256 // levels)
    lvl = (frame_rgb.astype(np.uint16) + (q // 2)) // q
    lvl = np.clip(lvl, 0, levels - 1)
    return (lvl * q + (q // 2)).clip(0, 255).astype(np.uint8)


def pack_color_id(rgb: np.ndarray) -> np.ndarray:
    """Квантованный RGB ``(H, W, 3)`` -> int32 id ``RRGGBB`` (дешёвое сравнение соседей)."""
    return ((rgb[:, :, 0].astype(np.int32) << 16)
            | (rgb[:, :, 1].astype(np.int32) << 8)
            | rgb[:, :, 2].astype(np.int32))


def delta_change_mask(color_ids: np.ndarray, emit: Optional[np.ndarray] = None) -> np.ndarray:
    """Булева маска позиций, где нужно поставить ANSI-префикс (дельта-кодирование).

    Префикс ставится на первой эмитируемой позиции строки и всякий раз, когда
    цвет отличается от предыдущей позиции. ``emit`` — доп. ограничение «где
    вообще красим» (например, только линии контура).
    """
    change = np.empty(color_ids.shape, dtype=bool)
    change[:, 0] = True
    change[:, 1:] = color_ids[:, 1:] != color_ids[:, :-1]
    if emit is not None:
        change &= emit
    return change


def build_prefix_map(color_ids: np.ndarray, change: np.ndarray) -> dict:
    """Словарь «id цвета -> ANSI-префикс» только по реально используемым цветам.

    Уникальных квантованных цветов обычно десятки, а не тысячи, поэтому карта
    мала и её построение стоит O(#уникальных), а не O(H*W) аллокаций строк.
    """
    uniq = np.unique(color_ids[change]).tolist()
    return {
        i: (" " if i < 0 else ansi_fg_truecolor((i >> 16) & 0xFF, (i >> 8) & 0xFF, i & 0xFF))
        for i in uniq
    }


# ---------------------------------------------------------------------------
# Абстракция подсветки
# ---------------------------------------------------------------------------

class SyntaxHighlighter(abc.ABC):
    """Абстрактный класс подсветки «синтаксиса» ASCII-вывода (ANSI-цвета).

    Наследники реализуют :meth:`apply` — сборку финального текста из строк
    сетки символов и цветов оригинала. Выбор конкретной стратегии — за
    высокоуровневым API (``fullcolor``/``edge_color``).
    """

    #: Печатать ли цветовые escape-коды вообще.
    enabled: bool = True

    @abc.abstractmethod
    def apply(self, rows: Sequence[str], frame_rgb: np.ndarray) -> str:
        """Собирает готовый многострочный текст из строк ``rows`` и кадра-источника цвета.

        Args:
            rows: текстовые строки кодовой сетки (по одной на ряд изображения).
            frame_rgb: кадр ``(h, w, 3)`` uint8 — источник цвета (RGB/BGR не
                важен: каналы трактуются как есть, порядок задаёт вызывающий).

        Returns:
            Готовая строка (возможно, с ANSI-кодами), переводы строк ``\\n``.
        """
        raise NotImplementedError


class NoHighlighter(SyntaxHighlighter):
    """Подсветка выключена: возвращает текст как есть (монохромный вывод)."""

    enabled = False

    def apply(self, rows: Sequence[str], frame_rgb: np.ndarray) -> str:  # noqa: D102
        return "\n".join(rows)


class TruecolorHighlighter(SyntaxHighlighter):
    """RGB truecolor с дельта-кодированием префиксов.

    Args:
        levels: уровней квантования на канал (обычно из
            :func:`~ascii_art_lib.core.palettes.palette_color_levels`).
        color_mask: необязательная булева маска ``(h, w)`` — позиции, которые
            вообще окрашиваются (например, только линии контура).
        edge_only: если задана ``color_mask`` и ``edge_only=True``, позиции вне
            маски принудительно печатаются БЕЗ цвета (сброс), т.е. окрашивается
            только то, что помечено маской.
        reset: последовательность сброса в конце каждой строки.
    """

    def __init__(
        self,
        levels: int = 32,
        *,
        color_mask: Optional[np.ndarray] = None,
        edge_only: bool = False,
        reset: str = ANSI_RESET,
    ) -> None:
        self.levels = max(1, min(256, int(levels)))
        self.color_mask = color_mask
        self.edge_only = edge_only
        self.reset = reset

    def apply(self, rows: Sequence[str], frame_rgb: np.ndarray) -> str:  # noqa: D102
        from ascii_art_lib.core.image_ops import resize_frame

        h = len(rows)
        w = max((len(r) for r in rows), default=0)
        if w == 0 or h == 0:
            return ""
        src = frame_rgb
        if src.ndim == 2:
            src = np.dstack([src] * 3)
        if src.shape[0] != h or src.shape[1] != w:
            src = resize_frame(src, (w, h))

        val = quantize_color(src, self.levels)
        cid = pack_color_id(val)

        mask = self.color_mask
        if mask is not None:
            if mask.shape != cid.shape:
                mask = mask[:, :w] if mask.shape[1] >= w else mask
            if self.edge_only:
                # Вне маски — «пустой» ключ: печать символа без цветового кода
                cid = np.where(mask, cid, -1).astype(np.int32)

        change = delta_change_mask(cid, mask if (mask is not None and not self.edge_only) else None)
        prefix_at = build_prefix_map(cid, change)

        cid_list = cid.tolist()
        chg_list = change.tolist()
        out_lines: List[str] = []
        for y in range(h):
            row = rows[y]
            row_cid = cid_list[y]
            row_ch = chg_list[y]
            parts: List[str] = []
            run = ""
            x = 0
            painted = False
            for ch in row:
                if x < w and row_ch[x]:
                    if run:
                        parts.append(run)
                    pfx = prefix_at[row_cid[x]]
                    run = pfx
                    painted = painted or pfx.startswith("\033")
                run += ch
                x += 1
            if run:
                parts.append(run)
            # Сброс ставим только если в строке реально были цветовые коды:
            # пустые/неокрашенные строки остаются ровно целевой ширины.
            if painted:
                parts.append(self.reset)
            out_lines.append("".join(parts))
        return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Абстракция рендера
# ---------------------------------------------------------------------------

class BaseRenderer(abc.ABC):
    """Абстрактный выводчик ASCII-текста (консоль, файл, NUL и т.п.).

    Потомки обязаны реализовать :meth:`show_static` (один кадр) и
    :meth:`play` (поток кадров); :meth:`save` имеет общую файловую реализацию.
    """

    @abc.abstractmethod
    def show_static(self, text: str, *, header: str = "") -> None:
        """Выводит один статичный ASCII-кадр."""
        raise NotImplementedError

    @abc.abstractmethod
    def play(
        self,
        frames: Iterable[str],
        fps: float,
        **kwargs: object,
    ) -> None:
        """Проигрывает поток готовых строк-кадров с частотой ``fps``."""
        raise NotImplementedError

    def save(self, text: str, path: str) -> str:
        """Сохраняет текст в файл (общая реализация); возвращает путь."""
        import os

        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path


class NullRenderer(BaseRenderer):
    """Рендер-заглушка: ничего не выводит (тихие прогоны, тесты, ``--no-print``)."""

    def show_static(self, text: str, *, header: str = "") -> None:  # noqa: D102
        return None

    def play(self, frames: Iterable[str], fps: float, **kwargs: object) -> None:  # noqa: D102
        for _ in frames:
            pass
        return None
