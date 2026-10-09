"""CLI: конвертация изображений/анимаций в ASCII-арт прямо из консоли.

Запуск после установки пакета (через entry point ``asciiart``)::

    asciiart photo.jpg --size 120x40 --save out.txt
    asciiart clip.gif --fps 24 --duration 5 --no-color

Или без установки, из исходников::

    python -m ascii_art_lib photo.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

from .api import convert_animation, convert_image, play_animation, save_ascii
from .media import probe
from .palettes import PALETTES, DEFAULT_PALETTE


def _parse_size(value: str) -> Optional[Tuple[int, int]]:
    """Разбирает 'WxH' / 'W,H' / 'W H' в кортеж; 'auto'/'-' -> None."""
    if not value or value.lower() in ("auto", "-"):
        return None
    for sep in ("x", "X", ",", " "):
        if sep in value:
            a, b = value.split(sep, 1)
            return int(a), int(b)
    raise argparse.ArgumentTypeError(f"Неверный формат размера: {value!r} (ожидается WxH)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="asciiart",
        description="Конвертер изображений и анимаций (GIF/видео) в ASCII-арт.",
    )
    p.add_argument("inputs", nargs="+", help="Файлы: изображения (png/jpg/webp…) или анимации (gif/mp4/avi…)")
    p.add_argument("--palette", "-p", default=DEFAULT_PALETTE,
                   help=f"Имя палитры ({', '.join(PALETTES)}) или своя строка символов.")
    p.add_argument("--reverse-palette", "-r", action="store_true", dest="reverse_palette",
                   help="Перевернуть палитру (свет/тень символами наоборот).")
    p.add_argument("--size", "-s", type=_parse_size, default=None, metavar="WxH",
                   help="Размер вывода в символах, напр. 120x40. По умолчанию — авто под терминал.")
    p.add_argument("--no-color", dest="color", action="store_false", default=True,
                   help="Монохромный режим (без ANSI-цветов).")
    p.add_argument("--color-levels", type=int, default=None, metavar="N",
                   help="Уровней квантования на цветовой канал. "
                        "По умолчанию выводится из размера палитры.")
    p.add_argument("--invert", "-i", action="store_true", help="Инвертировать яркость.")
    p.add_argument("--edges", nargs="?", const="canny", default=None, metavar="METHOD",
                   choices=["canny", "sobel"],
                   help="Выделение контуров перед конвертацией (canny по умолчанию либо sobel).")
    p.add_argument("--edge-mode", default="lines", choices=["lines", "overlay", "palette"],
                   help="Режим контуров: 'lines' — только линии (символы яркости), "
                        "'overlay' — поверх оригинала, "
                        "'palette' — собственная палитра ориентации контуров "
                        "(наклон линии -> '/ - \\ |', изгибы -> '^ v < > ( ) [ ] { }').")
    p.add_argument("--curve-threshold", type=float, default=0.5, metavar="X",
                   help="Чувствительность определения изгиба для --edge-mode palette "
                        "(меньше — больше скобочных символов; по умолчанию 0.5).")
    p.add_argument("--edge-low", type=int, default=50, metavar="N",
                   help="Нижний порог детекции контуров (по умолчанию 50).")
    p.add_argument("--edge-high", type=int, default=150, metavar="N",
                   help="Верхний порог детекции контуров (по умолчанию 150).")
    p.add_argument("--edge-blur", type=int, default=5, metavar="K",
                   help="Размер гауссова ядра перед детекцией контуров (0 — выключить).")
    p.add_argument("--max-pixels", type=int, default=32_000_000, metavar="PX",
                   help="Лимит площади входного кадра для экономии RAM (0 = без лимита).")
    p.add_argument("--fps", type=float, default=None,
                   help="Переопределить FPS при воспроизведении анимации.")
    p.add_argument("--duration", type=float, default=None,
                   help="Ограничить время проигрывания анимации, сек.")
    p.add_argument("--play", action="store_true",
                   help="Проиграть анимацию в терминале вместо пакетной конвертации.")
    p.add_argument("--save", "-o", default=None, metavar="PATH_OR_DIR",
                   help="Сохранить результат: файл (для картинок) или директорию кадров (для анимаций).")
    p.add_argument("--print", dest="do_print", action="store_true", default=None,
                   help="Печатать ASCII в stdout.")
    p.add_argument("--no-print", dest="do_print", action="store_false",
                   help="Не печатать (только сохранить).")
    p.add_argument("--progress", action="store_true", help="Показывать прогресс-бар конвертации.")
    p.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    return p


def _save_one(text: str, input_path: str, save_arg: str, index: int, total: int) -> str:
    """Выбирает имя файла результата, если --save указывает на директорию."""
    if save_arg.endswith((os.sep, "/")) or os.path.isdir(save_arg):
        os.makedirs(save_arg, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        suffix = f"_{index}" if total > 1 else ""
        return os.path.join(save_arg, f"{base}{suffix}.txt")
    if total > 1:
        root, ext = os.path.splitext(save_arg)
        return f"{root}_{index}{ext}"
    return save_arg


def run(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    max_pixels = args.max_pixels if args.max_pixels and args.max_pixels > 0 else None

    # Определяем тип каждого входа: картинка или анимация
    entries = []
    for path in args.inputs:
        if not os.path.isfile(path):
            print(f"error: файл не найден: {path}", file=sys.stderr)
            return 2
        entries.append((path, probe(path)))

    animations = [(p, i) for p, i in entries if i.kind == "animation"]
    images = [(p, i) for p, i in entries if i.kind == "image"]

    do_print = args.do_print if args.do_print is not None else (args.save is None)

    # Общие параметры конвертации (для картинок и анимаций)
    common = dict(
        palette=args.palette,
        reverse_palette=args.reverse_palette,
        size=args.size,
        fullcolor=args.color,
        color_levels=args.color_levels,
        invert=args.invert,
        max_pixels=max_pixels,
        edges=args.edges,
        edge_mode=args.edge_mode,
        low_threshold=args.edge_low,
        high_threshold=args.edge_high,
        blur_ksize=args.edge_blur,
        curve_threshold=args.curve_threshold,
    )

    rc = 0

    # ---- Анимации -----------------------------------------------------------
    for path, info in animations:
        try:
            if args.play:
                play_animation(
                    path,
                    fps=args.fps,
                    duration=args.duration,
                    save_dir=args.save if args.save else None,
                    progress=args.progress,
                    **common,
                )
            else:
                frames = convert_animation(path, progress=args.progress, **common)
                if do_print:
                    for text in frames:
                        sys.stdout.write("\033[H\033[J")
                        print(text, flush=True)
                        sys.stdout.flush()
                        if args.fps:
                            import time
                            time.sleep(1.0 / args.fps)
                    continue
                # Пакетная запись кадров в файлы
                out_dir = args.save or "."
                os.makedirs(out_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(path))[0]
                for i, text in enumerate(frames):
                    ext = "ans" if args.color else "txt"
                    save_ascii(text, os.path.join(out_dir, f"{base}_{i:05d}.{ext}"))
        except Exception as e:  # noqa: BLE001
            print(f"error processing {path!r}: {e}", file=sys.stderr)
            rc = 1

    # ---- Статичные изображения ----------------------------------------------
    n_img = len(images)
    for idx, (path, info) in enumerate(images, start=1):
        try:
            text = convert_image(path, **common)
            if do_print:
                print(text)
            if args.save:
                target = _save_one(text, path, args.save, idx, n_img + len(animations))
                save_ascii(text, target)
                print(f"saved: {target}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"error processing {path!r}: {e}", file=sys.stderr)
            rc = 1

    return rc


def main() -> None:  # entry point
    sys.exit(run())


if __name__ == "__main__":
    main()
