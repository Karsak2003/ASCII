"""Подпакет CLI: аргументический интерфейс над публичным API пакета.

* :mod:`~ascii_art_lib.cli.main` — парсер аргументов (включая отдельную
  «вкладку» справки по контурам ``--edges -h``) и точка входа ``main()``
  (entry point ``asciiart``, запуск ``python -m ascii_art_lib``,
  ``python main.py``).
"""

from __future__ import annotations

from .main import build_parser, main, run

__all__ = ["build_parser", "main", "run"]
