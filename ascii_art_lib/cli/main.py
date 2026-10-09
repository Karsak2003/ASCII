"""CLI: конвертация изображений/анимаций в ASCII-арт прямо из консоли.

Запуск после установки пакета (через entry point ``asciiart``)::

    asciiart photo.jpg --size 120x40 --save out.txt
    asciiart clip.gif --fps 24 --duration 5 --no-color

Или без установки, из исходников::

    python -m ascii_art_lib photo.jpg
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
from typing import List, Optional, Tuple

from ..api import convert_animation, convert_image, play_animation, save_ascii
from ..core.media import probe
from ..core.palettes import PALETTES, DEFAULT_PALETTE


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

#: Флаги, относящиеся к выделению контуров. Скрыты из общей справки, кроме
#: самого ``--edges``/``-e`` — он всегда виден в основной помощи.
EDGE_FLAGS = ("--edge-mode", "--curve-threshold",
              "--edge-low", "--edge-high", "--edge-blur",
              "--edge-overlay", "--no-edge-overlay",
              "--edge-fill", "--edge-color", "--no-edge-color")

#: Значение по умолчанию для ``--edge-fill`` (пустой фон контура).
DEFAULT_EDGE_FILL = "space"

#: Полные тексты help всех edge-флагов (реестр). Используется и при сборке
#: основной справки (--edges виден, остальные скрыты), и во вкладке --edges -h.
EDGE_FLAG_HELP = {
    "--edges": (
        "Выделение контуров: результат ВСЕГДА рисуется собственной палитрой "
        "ориентации — символ повторяет наклон линии ('/|-\\\\'), а изогнутые "
        "участки получают парные скобки ('^v<>()[]{}'). Без аргумента — canny. "
        "Все тонкие настройки контуров спрятаны в отдельной справке: "
        "запустите с '--edges -h'."
    ),
    #: Краткий вариант для ОСНОВНОЙ справки (--edges остаётся видимым);
    #: полная формулировка — во вкладке ``--edges -h`` (значение "--edges").
    "--edges_main": (
        "Выделение контуров (палитра ориентации применяется всегда). "
        "Тонкие настройки: запустите с '--edges -h'."
    ),
    "--edge-mode": (
        "Что делать с ИЗГИБАМИ контура (палитра ориентации применяется всегда; "
        "вкладка '--edges -h'): 'curves' (по умолчанию) — изогнутые участки "
        "получают скобочные символы '^ v < > ( ) [ ] { }'; 'lines' — только "
        "базовый набор наклона '/ - \\\\ |', без скобок; 'overlay' — контуры "
        "поверх оригинала (яркостная ASCII-конвертация обычной палитрой вместо "
        "палитры ориентации)."
    ),
    "--curve-threshold": (
        "--edge-mode curves: чувствительность определения изгиба "
        "(меньше — больше скобочных символов; по умолчанию 0.5)."
    ),
    "--edge-low": "Нижний порог детекции контуров (по умолчанию 50).",
    "--edge-high": "Верхний порог детекции контуров (по умолчанию 150).",
    "--edge-blur": (
        "Размер гауссова ядра перед детекцией контуров (0 — выключить)."
    ),
    "--edge-overlay": (
        "Наложение контура ПОВЕРХ изображения (отдельный флаг, по умолчанию "
        "НЕ накладывает): сначала строится обычная ASCII-картинка оригинала "
        "(--palette/--invert), затем в позициях линий её символы замещаются "
        "символами палитры ориентации. Отключить: --no-edge-overlay."
    ),
    "--edge-fill": (
        "Заполнение фона контура, когда наложение ВЫКЛЮЧЕНО (--edge-mode "
        "curves|lines): 'space' — пустой фон (по умолчанию); одиночный символ — "
        "однотонная канва из него (напр. '--edge-fill .', '--edge-fill \"#\"'); "
        "'brightness' — фон заполняется символами яркостной палитры (контуры "
        "поверх ASCII-изображения оригинала)."
    ),
    "--edge-color": (
        "Окрашивание контура ANSI-цветом оригинального кадра (по умолчанию "
        "включено, если не задан --no-color). Отключить: --no-edge-color — "
        "контур печатается без цветовых кодов даже в цветном режиме."
    ),
}


def _edge_fill_arg(value: str) -> str:
    """Валидатор ``--edge-fill``: 'space', 'brightness' или одиночный символ."""
    from ..edges.palette import is_edge_fill_valid

    if not value or not is_edge_fill_valid(value):
        raise argparse.ArgumentTypeError(
            f"Неверное заполнение фона: {value!r}. Используйте 'space', "
            "'brightness' или одиночный символ, напр. '.' или '#'."
        )
    return value


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
    """Прячет тонкие edge-флаги из основной справки — они живут во вкладке
    ``--edges -h``. Сам ``--edges`` остаётся видимым в общей помощи."""
    for action in p._actions:  # noqa: SLF001 (argparse не имеет публичного API)
        if any(opt in EDGE_FLAGS for opt in action.option_strings):
            action.help = argparse.SUPPRESS


#: Текст-описание вкладки контуров (печатается до списка флагов).
_EDGE_HELP_INTRO = (
    "Когда включён --edges, результат ВСЕГДА рисуется собственной палитрой\n"
    "ориентации контуров: символ повторяет наклон линии ('/ - \\ |'), а изогнутые\n"
    "участки получают парные скобки ('^ v < > ( ) [ ] { }').\n"
    "Флаг --edge-mode нужен только для управления ИЗГИБАМИ (скобочными символами)\n"
    "и режимом наложения — сама палитра ориентации применяется всегда.\n"
    "\n"
    "Примеры:\n"
    "  asciiart photo.png --edges\n"
    "  asciiart photo.png -e sobel --edge-low 30 --edge-high 100\n"
    "  asciiart photo.png --edges --edge-mode lines          # без скобок\n"
    "  asciiart photo.png --edges --edge-mode curves --curve-threshold 0.3\n"
    "  asciiart photo.png --edges --edge-mode overlay        # контуры поверх оригинала\n"
    "  asciiart photo.png --edges --edge-overlay             # контур ПОВЕРХ ASCII-картинки\n"
    "  asciiart photo.png --edges --edge-fill .              # точечный фон вместо пустого\n"
    "  asciiart photo.png --edges --edge-fill brightness     # яркостная канва под контуром\n"
    "  asciiart photo.png --edges --no-edge-color            # контур без ANSI-цветов\n"
    "  asciiart photo.png --edges --edge-mode lines --edge-fill ':'\n"
)


def print_edge_help(p: Optional[argparse.ArgumentParser] = None) -> None:
    """Печатает отдельную справку («вкладку») по флагам выделения контуров.

    Реализовано только на публичном API :mod:`argparse` (``add_help=False`` +
    стандартный ``format_help()``), поэтому не зависит от приватных атрибутов
    форматтера и не падает между версиями Python.
    """
    p = p or build_parser()
    q = argparse.ArgumentParser(
        prog="EDGE — ФЛАГИ ВЫДЕЛЕНИЯ КОНТУРОВ (asciiart --edges -h)",
        description=_EDGE_HELP_INTRO,
        epilog="Открыть эту вкладку:  asciiart --edges -h   (или коротко: -e -h)",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for action in p._actions:  # noqa: SLF001 (только чтение списка действий)
        opts = action.option_strings
        if "--edges" in opts or any(opt in EDGE_FLAGS for opt in opts):
            clone = copy.copy(action)
            clone.help = EDGE_FLAG_HELP.get(opts[-1], _restore_suppressed_help(action))
            q._add_action(clone)  # noqa: SLF001 (стабильный внутренний метод argparse)
    print(q.format_help())


def _restore_suppressed_help(action: argparse.Action) -> Optional[str]:
    """Возвращает help-текст флага даже если он был скрыт через SUPPRESS.

    Основная справка скрывает edge-флаги (``action.help = SUPPRESS``), но их
    тексты нужны во вкладке ``--edges -h`` — восстанавливаем их из реестра
    :data:`EDGE_FLAG_HELP`.
    """
    if action.help != argparse.SUPPRESS and action.help is not None:
        return action.help
    for opt in action.option_strings:
        if opt in EDGE_FLAG_HELP:
            return EDGE_FLAG_HELP[opt]
    return action.help


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
                   help=EDGE_FLAG_HELP["--edges_main"])
    p.add_argument("--edge-mode", default="curves", choices=["curves", "lines", "overlay"],
                   help=EDGE_FLAG_HELP["--edge-mode"])
    p.add_argument("--curve-threshold", type=float, default=0.5, metavar="X",
                   help=EDGE_FLAG_HELP["--curve-threshold"])
    p.add_argument("--edge-low", type=int, default=50, metavar="N",
                   help=EDGE_FLAG_HELP["--edge-low"])
    p.add_argument("--edge-high", type=int, default=150, metavar="N",
                   help=EDGE_FLAG_HELP["--edge-high"])
    p.add_argument("--edge-blur", type=int, default=5, metavar="K",
                   help=EDGE_FLAG_HELP["--edge-blur"])
    # Наложение контура поверх изображения — отдельный флаг, по умолчанию НЕ накладывает
    p.add_argument("--edge-overlay", dest="edge_overlay", action="store_true",
                   default=False, help=EDGE_FLAG_HELP["--edge-overlay"])
    p.add_argument("--no-edge-overlay", dest="edge_overlay", action="store_false",
                   help=argparse.SUPPRESS)
    # Заполнение фона при выключенном наложении: space / одиночный символ / brightness
    p.add_argument("--edge-fill", type=_edge_fill_arg, default=DEFAULT_EDGE_FILL,
                   metavar="SYM", help=EDGE_FLAG_HELP["--edge-fill"])
    # Окрашивание контура (по умолчанию — авто: цвет, если включён fullcolor)
    p.add_argument("--edge-color", dest="edge_color", action="store_true",
                   default=None, help=EDGE_FLAG_HELP["--edge-color"])
    p.add_argument("--no-edge-color", dest="edge_color", action="store_false",
                   help=argparse.SUPPRESS)
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


def _common_kwargs(args: argparse.Namespace) -> dict:
    """Собирает единый словарь параметров конвертации для картинок и анимаций."""
    max_pixels = args.max_pixels if args.max_pixels and args.max_pixels > 0 else None
    return dict(
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
        edge_overlay=args.edge_overlay,
        edge_fill=args.edge_fill,
        edge_color=args.edge_color,
    )


def _classify_inputs(paths: List[str]) -> Tuple[List[Tuple[str, object]],
                                                 List[Tuple[str, object]], int]:
    """Делит входы на статичные изображения и анимации; (images, animations, rc)."""
    images: List[Tuple[str, object]] = []
    animations: List[Tuple[str, object]] = []
    for path in paths:
        if not os.path.isfile(path):
            print(f"error: файл не найден: {path}", file=sys.stderr)
            return images, animations, 2
        info = probe(path)
        (animations if info.kind == "animation" else images).append((path, info))
    return images, animations, 0


def _handle_animation(path: str, args: argparse.Namespace, common: dict,
                      do_print: bool) -> None:
    """Один анимационный вход: проигрывание, потоковый вывод или пакетная запись кадров."""
    if args.play:
        play_animation(
            path,
            fps=args.fps,
            duration=args.duration,
            save_dir=args.save if args.save else None,
            progress=args.progress,
            **common,
        )
        return
    frames = convert_animation(path, progress=args.progress, **common)
    if do_print:
        import time

        for text in frames:
            sys.stdout.write("\033[H\033[J")
            print(text, flush=True)
            sys.stdout.flush()
            if args.fps:
                time.sleep(1.0 / args.fps)
        return
    out_dir = args.save or "."
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    for i, text in enumerate(frames):
        ext = "ans" if args.color else "txt"
        save_ascii(text, os.path.join(out_dir, f"{base}_{i:05d}.{ext}"))


def _handle_image(path: str, args: argparse.Namespace, common: dict,
                  do_print: bool, idx: int, total: int) -> None:
    """Одно статичное изображение: конвертация + печать/сохранение."""
    text = convert_image(path, **common)
    if do_print:
        print(text)
    if args.save:
        target = _save_one(text, path, args.save, idx, total)
        save_ascii(text, target)
        print(f"saved: {target}", file=sys.stderr)


def run(argv: Optional[List[str]] = None) -> int:
    """Точка входа CLI: разбор аргументов и диспетчер по типам входов."""
    raw = list(sys.argv[1:] if argv is None else argv)
    # Вкладка «--edges -h»: отдельная справка по контурам, файлы не нужны
    if any(a in ("--edges", "-e") for a in raw) and any(a in ("-h", "--help") for a in raw):
        print_edge_help()
        return 0
    p = build_parser()
    # Запоминаем аргументы на парсере — нужно для вкладки «--edges -h»
    p._asciiart_argv = raw  # noqa: SLF001
    args = p.parse_args(argv)

    images, animations, rc = _classify_inputs(args.inputs)
    if rc:
        return rc

    do_print = args.do_print if args.do_print is not None else (args.save is None)
    common = _common_kwargs(args)

    for path, _info in animations:
        try:
            _handle_animation(path, args, common, do_print)
        except Exception as e:  # noqa: BLE001
            print(f"error processing {path!r}: {e}", file=sys.stderr)
            rc = 1

    total = len(images) + len(animations)
    for idx, (path, _info) in enumerate(images, start=1):
        try:
            _handle_image(path, args, common, do_print, idx, total)
        except Exception as e:  # noqa: BLE001
            print(f"error processing {path!r}: {e}", file=sys.stderr)
            rc = 1

    return rc


def main() -> None:  # entry point
    sys.exit(run())


if __name__ == "__main__":
    main()
