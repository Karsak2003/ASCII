"""Класс вывода ASCII-арта в консоль (статика и анимация).

Отделяет «как показать» от «что показать»: конвертеры возвращают готовые строки,
а :class:`ConsoleRenderer` отвечает за очистку экрана, позиционирование курсора,
FPS-циклы анимации и запись в файл.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from typing import Callable, Iterator, Optional, TextIO

# Escape-последовательности ANSI (кроссплатформенно: Windows 10+ поддерживает их сам)
_CLEAR_SCREEN = "\033[H\033[J"
_CLEAR_SCROLLBACK = "\033[H\033[3J"
_HOME = "\033[H"


def _enable_ansi_on_windows() -> None:
    """Включает VT-обработку на консолях Windows (legacy консолям это нужно)."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


class ConsoleRenderer:
    """Класс вывода изображения (ASCII-строк) в консоль.

    Args:
        width/height: Принудительный размер области вывода в символах;
            по умолчанию — размер терминала.
        keep_aspect: Сохранять пропорции источника (компенсирует прямоугольность
            символов терминала множителем ~2 по ширине).
        clear_scrollback: Очищать историю прокрутки между кадрами (меньше «мусора»).
        out: Потоковый вывод (по умолчанию ``sys.stdout``).
    """

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        *,
        keep_aspect: bool = True,
        clear_scrollback: bool = False,
        out: Optional[TextIO] = None,
    ) -> None:
        _enable_ansi_on_windows()
        term_w, term_h = shutil.get_terminal_size((80, 24))
        self.width = int(width or term_w)
        self.height = int(height or term_h)
        self.keep_aspect = keep_aspect
        self.clear_scrollback = clear_scrollback
        self.out = out or sys.stdout

    # ------------------------------------------------------------------ utils
    def fit_size(self, src_w: int, src_h: int) -> tuple:
        """Размер ``(w, h)`` области вывода с сохранением пропорций источника.

        Компенсирует прямоугольную форму символов (символ примерно в 2 раза выше,
        чем широк): ширина считается как ``2 * h / aspect``.
        """
        if src_w <= 0 or src_h <= 0:
            return self.width, self.height
        aspect = src_h / src_w
        if self.keep_aspect:
            # Отступ сверху под служебную строку статуса
            usable_h = max(1, self.height - 1)
            # Компенсация пропорций символа: множитель 2 по горизонтали
            w = min(self.width, int(2.0 * usable_h / aspect))
            h = max(1, min(usable_h, int(w * aspect / 2.0)))
            return max(2, w), h
        return self.width, self.height - 1

    # ----------------------------------------------------------------- output
    def clear(self) -> None:
        seq = _CLEAR_SCREEN + (_CLEAR_SCROLLBACK if self.clear_scrollback else "")
        self.out.write(seq)
        self.out.flush()

    def show_static(self, text: str, *, header: str = "") -> None:
        """Выводит один статичный ASCII-кадр (с необязательной шапкой-статусом)."""
        self.clear()
        if header:
            self.out.write(header + "\n")
        self.out.write(text)
        if not text.endswith("\n"):
            self.out.write("\n")
        self.out.flush()

    def play(
        self,
        frames: Iterator[str],
        fps: float,
        *,
        duration: Optional[float] = None,
        header_fn: Optional[Callable[[int, int], str]] = None,
        stop_key_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Проигрывает генератор готовых строк-кадров с заданным FPS.

        Кадры потребляются **лениво** из генератора, поэтому анимация любой длины
        не требует хранения всех кадров в памяти.

        Args:
            frames: Итератор строк (уже сконвертированного ASCII-арта).
            fps: Желаемая частота кадров.
            duration: Ограничение по времени (секунды); ``None`` — до конца потока.
            header_fn: ``f(frame_index, loop_index) -> str`` для строки статуса.
            stop_key_check: Функция, возвращающая True для досрочного выхода.
        """
        frame_time = 1.0 / max(fps, 1e-6)
        start: Optional[float] = None
        idx = 0

        for text in frames:
            now = time.perf_counter()
            if start is None:
                start = now
            delay = start + idx * frame_time - now
            if delay > 0:
                time.sleep(delay)

            self.out.write(_HOME if idx else _CLEAR_SCREEN)
            if header_fn is not None:
                self.out.write(header_fn(idx, 0) + "\n")
            self.out.write(text)
            self.out.flush()

            idx += 1
            if duration is not None and (time.perf_counter() - start) >= duration:
                break
            if stop_key_check is not None and stop_key_check():
                break

    def save(self, text: str, path: str) -> None:
        """Сохраняет ASCII/ANSI-строку в файл (режим ANS/SRT-совместимого вывода)."""
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
