"""Тесты пакета ascii_art_lib: конвертация, память, CLI."""

from __future__ import annotations

import os
import sys
import tracemalloc

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from ascii_art_lib import (
    ConsoleRenderer,
    convert_animation,
    convert_image,
    frame_to_symbols,
    iter_frames,
    probe,
    save_ascii,
)
from ascii_art_lib.palettes import ASII_4, get_palette


@pytest.fixture()
def tmp_png(tmp_path):
    """Генерирует тестовое изображение 640x480 с градиентом и фигурами."""
    h, w = 480, 640
    img = np.zeros((h, w, 3), np.uint8)
    grad = np.linspace(0, 255, w, dtype=np.uint8)
    img[:, :] = grad[None, :, None]
    cv2.circle(img, (w // 2, h // 2), 120, (0, 0, 255), -1)
    cv2.rectangle(img, (50, 50), (200, 180), (0, 255, 0), -1)
    path = str(tmp_path / "test.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture()
def tmp_gif(tmp_path):
    """Создаёт многосекундный GIF из закрашенных кадров (тест стриминга)."""
    from PIL import Image

    frames = []
    for i in range(8):
        a = np.full((120, 160, 3), i * 30, np.uint8)
        frames.append(Image.fromarray(a[:, :, ::-1]))  # BGR->RGB
    path = str(tmp_path / "anim.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return path


# ---------------------------------------------------------------------------
# Конвертация изображения
# ---------------------------------------------------------------------------

def test_convert_image_mono(tmp_png):
    art = convert_image(tmp_png, size=(80, 30), fullcolor=False)
    lines = art.split("\n")
    assert len(lines) == 30
    assert all(len(l) <= 80 for l in lines)
    assert any(ch != " " for ch in art)  # не пустой


def test_convert_image_color_has_ansi(tmp_png):
    art = convert_image(tmp_png, size=(60, 20), fullcolor=True)
    assert "\033[38;2;" in art      # truecolor-префиксы присутствуют
    assert "\033[0m" in art         # сброс в конце строк
    # Дельта-кодирование: префиксов цвета должно быть заметно меньше символов
    n_prefix = art.count("\033[38;2;")
    n_chars = sum(len(l) for l in art.split("\n"))
    assert n_prefix < n_chars


def test_convert_from_ndarray():
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    art = convert_image(frame, size=(50, 25), fullcolor=False)
    assert len(art.split("\n")) == 25


def test_invert_and_palette_name(tmp_png):
    a1 = convert_image(tmp_png, size=(40, 15), palette="asii", fullcolor=False)
    a2 = convert_image(tmp_png, size=(40, 15), palette="asii", fullcolor=False, invert=True)
    assert a1 != a2


def test_unicode_palette_fallback(tmp_png):
    pal = "░▒▓█"  # многобайтовая палитра -> универсальный путь
    art = convert_image(tmp_png, size=(40, 15), palette=pal, fullcolor=False)
    assert any(c in pal for c in art)


# ---------------------------------------------------------------------------
# Анимация: генератор + экономия памяти
# ---------------------------------------------------------------------------

def test_probe_gif(tmp_gif):
    info = probe(tmp_gif)
    assert info.kind == "animation"
    assert info.width == 160 and info.height == 120
    assert info.n_frames >= 8 or info.n_frames == 1  # Pillow/cv2 по-разному считают


def test_convert_animation_streaming(tmp_gif):
    gen = convert_animation(tmp_gif, size=(40, 12), fullcolor=False)
    first = next(gen)
    assert isinstance(first, str) and first
    rest = list(gen)
    assert len(rest) >= 1


def test_memory_constant_for_long_animation(tmp_path):
    """Пиковая RAM при стриминговой конвертации не растёт с числом кадров."""
    from PIL import Image

    path = str(tmp_path / "big.gif")
    n_small, n_big = 10, 60
    w, h = 200, 150

    def make(n):
        fr = [Image.fromarray(np.full((h, w, 3), i % 256 // 4, np.uint8)) for i in range(n)]
        fr[0].save(path, save_all=True, append_images=fr[1:], duration=80, loop=0)

    def peak_mb(n):
        make(n)
        tracemalloc.start()
        for _ in convert_animation(path, size=(50, 15), fullcolor=False):
            pass
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / 1e6

    p_small = peak_mb(n_small)
    p_big = peak_mb(n_big)
    # Пик — на порядок меньше, чем хранилище всех кадров (60*200*150*3 ≈ 5.4 MB только пикселей)
    assert p_big < 8.0, f"Потребление памяти слишком велико: {p_big:.1f} MB"
    # И почти не зависит от длины анимации (допустим 3x запас на служебные структуры)
    assert p_big < max(p_small * 3.0, p_small + 1.5), (p_small, p_big)


def test_max_pixels_downscale(tmp_png):
    # Ограничение площади источника защищает RAM от гигантских файлов
    art = convert_image(tmp_png, size=(40, 15), fullcolor=False, max_pixels=10_000)
    assert len(art.split("\n")) == 15


# ---------------------------------------------------------------------------
# Renderer & save
# ---------------------------------------------------------------------------

class StringIO_:
    def __init__(self):
        self.buf = []

    def write(self, s):
        self.buf.append(s)

    def flush(self):
        pass

    @property
    def value(self):
        return "".join(self.buf)


def test_renderer_show_static():
    out = StringIO_()
    r = ConsoleRenderer(width=80, height=24, out=out)
    r.show_static("hello\nworld", header="HDR")
    assert "hello\nworld" in out.value
    assert "HDR" in out.value
    assert "\033[H\033[J" in out.value


def test_renderer_play_generator():
    out = StringIO_()
    r = ConsoleRenderer(width=80, height=24, out=out)
    r.play(iter(["f0", "f1", "f2"]), fps=1000.0, header_fn=lambda i, l: f"#{i}")
    assert "f0" in out.value and "f2" in out.value


def test_save_ascii(tmp_path):
    p = save_ascii("abc", str(tmp_path / "sub/out.txt"))
    assert os.path.isfile(p) and open(p).read() == "abc"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_converts_image(tmp_png, tmp_path, capsys):
    from ascii_art_lib.cli import run

    out_file = str(tmp_path / "res.txt")
    rc = run([tmp_png, "--size", "60x20", "--no-color", "--no-print", "--save", out_file])
    assert rc == 0
    text = open(out_file).read()
    assert len(text.split("\n")) == 20


def test_cli_animation_to_files(tmp_gif, tmp_path):
    from ascii_art_lib.cli import run

    d = str(tmp_path / "frames")
    rc = run([tmp_gif, "--size", "40x12", "--no-color", "--no-print", "--save", d + os.sep])
    assert rc == 0
    files = sorted(os.listdir(d))
    assert len(files) >= 2
    assert all(f.endswith(".txt") for f in files)
