"""Подпакет вывода: консольный рендер ASCII-арта.

* :class:`~ascii_art_lib.rendering.console.ConsoleRenderer` — реализация
  абстракции :class:`~ascii_art_lib.utils.ansi.BaseRenderer` для терминала
  (ANSI-очистка, позиционирование курсора, FPS-цикл анимации).
"""

from __future__ import annotations

from .console import ConsoleRenderer

__all__ = ["ConsoleRenderer"]
