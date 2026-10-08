"""Тонкая CLI-обёртка для запуска проекта простым запуском из консоли.

Эквивалентна ``python -m ascii_art_lib`` и позволяет запускать проект
без установки пакета:

    python main.py photo.jpg --size 120x40
    python main.py clip.gif --play --fps 24 --duration 5
"""

from ascii_art_lib.cli import main

if __name__ == "__main__":
    main()
