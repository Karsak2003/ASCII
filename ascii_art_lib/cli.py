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
import shutil
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


# ---------------------------------------------------------------------------
# «Вкладка» контуров: --edges -h печатает отдельную справку по edge-флагам
# ---------------------------------------------------------------------------

#: Флаги, относящиеся к выделению контуров (скрыты из общей справки).
EDGE_FLAGS = ("--edges", "--edge-mode", "--curve-threshold",
              "--edge-low", "--edge-high", "--edge-blur")


class _EdgesHelpAction(argparse.Action):
    """Открывает отдельную «вкладку» справки по контурам.

    Срабатывает на комбинацию ``--edges -h`` (короткий вариант ``-e -h``):
    если вместе с ``--edges`` в аргументах есть ``-h``/``--help``, печатается
    только раздел EDGE и программа завершается. Сам флаг ``--edges`` при этом
    продолжает работать как обычно (если ``-h`` не указан).
    """

    def __init__(self, option_strings, dest, help=None, **kwargs):  # noqa: D107
        super().__init__(option_strings, dest, nargs="?", const="canny",
                         default=None, help=help, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        argv = list(getattr(parser, "_asciiart_argv", []) or sys.argv[1:])
        if "-h" in argv or "--help" in argv:
            print_edge_help(parser)
            parser.exit(0)
        setattr(namespace, self.dest, values if values is not None else self.const)


def _hide_edge_flags(p: argparse.ArgumentParser) -> None:
    """Прячет edge-флаги из основной справки — они живут во вкладке ``--edges -h``."""
    for action in p._actions:  # noqa: SLF001 (argparse не имеет публичного API)
        if any(opt in EDGE_FLAGS for opt in action.option_strings):
            action.help = argparse.SUPPRESS


def print_edge_help(p: Optional[argparse.ArgumentParser] = None) -> None:
    """Печатает отдельную справку («вкладку») по флагам выделения контуров."""
    p = p or build_parser()
    width = shutil.get_terminal_size((80, 24)).columns
    fmt = argparse.HelpFormatter("asciiart [ФЛАГИ КОНТУРОВ]", max_help_position=30,
                                 width=max(60, min(width, 100)))
    print("EDGE — выделение контуров (отдельная справка; открыть: --edges -h)")
    print("=" * 72)
    print(fmt._format_text(  # noqa: SLF001
        "Когда включён ``--edges``, результат ВСЕГДА рисуется собственной "
        "палитрой ориентации контуров: символ повторяет наклон линии "
        "('/ - \\ |'), а изогнутые участки получают парные скобки "
        "('^ v < > ( ) [ ] { }'). Флаги ниже тонко настраивают этот режим.\n\n"
        "Примеры:\n"
        "  asciiart photo.png --edges\n"
        "  asciiart photo.png -e sobel --edge-low 30 --edge-high 100\n"
        "  asciiart photo.png --edges --edge-mode lines          # без скобок\n"
        "  asciiart photo.png --edges --edge-mode curves --curve-threshold 0.3\n"
        "  asciiart photo.png --edges --edge-mode overlay        # контуры поверх оригинала\n"
    ))
    dummy = argparse.ArgumentParser(add_help=False)
    group = dummy.add_argument_group("флаги контуров")  # _ArgumentGroup с публичным API
    for action in p._actions:  # noqa: SLF001
        if any(opt in EDGE_FLAGS for opt in action.option_strings):
            group._add_action(action)  # noqa: SLF001
    fmt._add_argument_group(group)  # noqa: SLF001 (приватный API argparse)
    print(fmt.format_help())


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
    p.add_argument("-e", "--edges", action=_EdgesHelpAction, metavar="METHOD",
                   choices=["canny", "sobel"],
                   help="Выделение контуров: результат всегда рисуется собственной "
                        "палитрой ориентации ('/|-\\' по наклону, изгибы — скобками). "
                        "Без аргумента — canny. Все тонкие настройки контуров скрыты "
                        "в отдельной справке: запустите с '--edges -h'.")
    p.add_argument("--edge-mode", default="curves", choices=["curves", "lines", "overlay"],
                   help="Что делать с изгибами контура (вкладка '--edges -h'): "
                        "'curves' (по умолчанию) — изогнутые участки получают "
                        "скобочные символы '^ v < > ( ) [ ] { }'; "
                        "'lines' — только базовый набор наклона '/ - \\ |', без скобок; "
                        "'overlay' — контуры поверх оригинала (яркостная ASCII-конвертация "
                        "обычной палитрой вместо палитры ориентации).")
    p.add_argument("--curve-threshold", type=float, default=0.5, metavar="X",
                   help="--edge-mode curves: чувствительность определения изгиба "
                        "(меньше — больше скобочных символов; по умолчанию 0.5).")
    p.add_argument("--edge-low", type=int, default=50, metavar="N",
                   help="Нижний порог детекции контуров (по умолчанию 50).")
    p.add_argument("--edge-high", type=int, default=150, metavar="N",
                   help="Верхний порог детекции контуров (по умолчанию 150).")
    p.add_argument("--edge-blur", type=int, default=5, metavar="K",
                   help="Размер гауссова ядра перед детекцией контуров (0 — выключить).")
    _hide_edge_flags(p)
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
    raw = list(sys.argv[1:] if argv is None else argv)
    # Вкладка «--edges -h»: отдельная справка по контурам, файлы не нужны
    if any(a in ("--edges", "-e") for a in raw) and any(a in ("-h", "--help") for a in raw):
        print_edge_help()
        return 0
    p = build_parser()
    # Запоминаем аргументы на парсере — нужно для вкладки «--edges -h»
    p._asciiart_argv = raw  # noqa: SLF001
    args = p.parse_args(argv)
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
