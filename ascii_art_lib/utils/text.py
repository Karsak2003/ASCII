"""Сборка текстовых строк из кодовых сеток (общие куски для api/edges).

``uint8``-сетка ``(h, w)`` -> список построчных строк текста — единственная
реализация на весь пакет (раньше дублировалась в ``api._grid_rows`` и
``edge_palette.decode_grid``).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

__all__ = ["grid_rows", "rows_to_text"]


def grid_rows(grid: np.ndarray) -> List[str]:
    """Полноширинная байтовая сетка ``(h, w)`` uint8 -> список строк (UTF-8 decode).

    Каждая строка содержит ровно ``w`` байтов; многобайтовые Unicode-символы,
    развёрнутые в UTF-8 внутри сетки, декодируются корректно, поэтому ширина
    вывода всегда равна целевой — пропорции ASCII-картинки сохраняются при
    любом смешивании слоёв (контур + канва + заполнение).
    """
    return [bytes(row).decode("utf-8", "replace") for row in np.asarray(grid, dtype=np.uint8)]


def rows_to_text(rows: Sequence[str]) -> str:
    """Список строк -> готовый многострочный текст."""
    return "\n".join(rows)
