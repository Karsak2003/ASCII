# ascii-art-lib 🖼️➡️🔤

Быстрая конвертация **изображений и анимаций (GIF / видео)** в цветной **ASCII-арт** для консоли.

Проект работает в двух режимах:

* **как Python-пакет** — импортируете функции `convert_image` / `convert_animation` и т.д.;
* **как CLI-утилита** — `python main.py …`, `python -m ascii_art_lib …` или `asciiart …` после установки.

Ключевые особенности:

- 🎨 Цветной вывод ANSI truecolor с квантованием цвета и дельта-кодированием escape-последовательностей;
- 🌀 Выделение контуров (Canny / Sobel) — можно использовать совместно с любой конвертацией;
- 📦 Экономия RAM: анимации читаются **покадрово (стриминг)**, память `O(1 кадр)` независимо от длительности файла + защитный лимит `max_pixels` для гигантских изображений;
- ⚡ Производительность: полностью векторные операции NumPy/OpenCV (LUT-таблицы вместо `np.vectorize`), обработка идёт на уменьшенном до целевого размера кадре;
- 🖥️ Класс `ConsoleRenderer` — отдельный вывод статики и анимации в терминал (с автоподбором размера под окно);
- 🔁 Переворачиваемые палитры и автоматический цветовой охват из размера палитры.

---

## Оглавление

1. [Установка](#установка)
2. [Быстрый старт (CLI)](#быстрый-старт-cli)
3. [Использование как пакет](#использование-как-пакет)
4. [Полное описание API](#полное-описание-api)
5. [Палитры символов](#палитры-символов)
6. [Выделение контуров](#выделение-контуров)
7. [Цветовой охват и палитра](#цветовой-охват-и-палитра)
8. [Оптимизация памяти и производительности](#оптимизация-памяти-и-производительности)
9. [Структура проекта](#структура-проекта)
10. [Тесты](#тесты)

---

## Установка

Требования: Python ≥ 3.9, `numpy`, `opencv-python`, `Pillow` (`progress` — опционально, для прогресс-бара).

```bash
# зависимости
pip install numpy opencv-python Pillow progress

# установка пакета (даёт команду asciiart в PATH)
pip install .

# либо editable-режим для разработки
pip install -e .[dev]
```

Запуск без установки тоже работает — из корня проекта:

```bash
python main.py photo.png            # тонкая обёртка над CLI
python -m ascii_art_lib photo.png   # то же самое через модуль
```

---

## Быстрый старт (CLI)

```bash
# картинка -> ASCII в консоль (размер авто-под терминал)
asciiart photo.png

# сохранить в файл
asciiart photo.png -o out.txt

# Задать размер в символах, своя палитра, инверсия
asciiart photo.png -s 120x40 -p asii_4 --invert

# перевёрнутая палитра (свет/тень символами наоборот)
asciiart photo.png --reverse-palette

# монохром
asciiart photo.png --no-color

# только контуры изображения (edge-art)
asciiart photo.png --edges
asciiart photo.png --edges sobel --edge-low 30 --edge-high 100

# контуры поверх оригинала
asciiart photo.png --edges --edge-mode overlay

# собственная палитра контуров: символ повторяет НАКЛОН линии (/ - \ |),
# изогнутые участки получают парные скобки (^ v < > ( ) [ ] { })
asciiart photo.png --edges --edge-mode palette --no-color
asciiart photo.png --edges --edge-mode palette --curve-threshold 0.3

# GIF/видео: проиграть в терминале
asciiart anim.gif --play
asciiart clip.mp4 --play --fps 15 --duration 10

# GIF/видео: сконвертировать все кадры в директорию
asciiart anim.gif --save frames/ --progress

# несколько файлов сразу
asciiart a.png b.jpg c.gif --save out_dir/
```

### Все флаги CLI

| Флаг | Описание |
|---|---|
| `inputs` (позиционный) | Один или несколько файлов: png/jpg/webp/bmp/tiff, gif/mp4/avi/mov/mkv/webm |
| `-p, --palette NAME` | Имя палитры (`asii`, `asii_1`, `asii_2`, `asii_3`, `asii_3v`, `asii_4`) или своя строка символов от тёмных к светлым |
| `-r, --reverse-palette` | Перевернуть палитру |
| `-s, --size WxH` | Размер вывода в символах (напр. `120x40`); по умолчанию — автоподбор под терминал с сохранением пропорций |
| `--no-color` | Монохромный режим (без ANSI-цветов) |
| `--color-levels N` | Уровней квантования на канал; по умолчанию выводится из размера палитры |
| `-i, --invert` | Инвертировать яркость |
| `--edges [canny\|sobel]` | Выделение контуров перед конвертацией (без аргумента — `canny`) |
| `--edge-mode {lines,overlay,palette}` | `lines` — только линии, `overlay` — контуры поверх оригинала, `palette` — палитра ориентации контуров (символ = наклон линии) |
| `--curve-threshold X` | Чувствительность определения изгиба для `--edge-mode palette` (0.5; меньше — больше скобочных символов) |
| `--edge-low N` / `--edge-high N` | Пороги двойной фильтрации контуров (50 / 150) |
| `--edge-blur K` | Гауссово размытие перед детекцией (5; `0` — выключить) |
| `--max-pixels PX` | Лимит площади входного кадра для защиты RAM (32 000 000; `0` — без лимита) |
| `--fps N` | Переопределить FPS при воспроизведении анимации |
| `--duration SEC` | Ограничить время проигрывания |
| `--play` | Проиграть анимацию в терминале вместо пакетной конвертации |
| `-o, --save PATH_OR_DIR` | Файл (для картинок) или директория кадров (для анимаций) |
| `--print` / `--no-print` | Печатать / не печатать результат в stdout |
| `--progress` | Прогресс-бар конвертации кадров |
| `--version` | Версия |

---

## Использование как пакет

```python
from ascii_art_lib import convert_image, show_image, play_animation, save_ascii

# 1) просто получить строку ASCII-арта
text = convert_image("photo.png", size=(100, 40))
print(text)

# 2) конвертация + вывод в консоль одной функцией
show_image("photo.png", palette="asii_4", invert=True)

# 3) сохранить результат (.txt — монохром, .ans — с ANSI-кодами)
save_ascii(convert_image("photo.png", fullcolor=False), "out.txt")

# 4) анимация — ленивый генератор ASCII-кадров (память O(1 кадра))
for frame_text in convert_animation("anim.gif"):
    print(frame_text)

# 5) проигрывание GIF/видео прямо в терминале
play_animation("clip.mp4", fps=20, duration=15, edges=True)

# 6) контуры + цвет + перевёрнутая палитра — всё вместе
show_image("photo.png", edges="canny", edge_mode="overlay",
           palette="asii", reverse_palette=True)
```

Результат `convert_image` — обычная `str`; при `fullcolor=True` она содержит ANSI truecolor-коды, поэтому корректно отображается в терминале и сохраняется в `.ans`-файл для последующего `cat`.

Можно передавать уже загруженный кадр (`np.ndarray` BGR, как из `cv2.imread`) — полезно, если изображение получено из камеры или другого конвейера:

```python
import cv2
from ascii_art_lib import convert_image

frame = cv2.imread("photo.png")          # ndarray (H, W, 3) BGR
print(convert_image(frame, size=(80, 30)))
```

---

## Полное описание API

### `convert_image(source, *, ...) -> str`

Конвертирует изображение (путь или `ndarray`) в ASCII-строку.

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `source` | `str \| np.ndarray` | — | Путь к файлу или BGR-кадр `(H, W, 3)` uint8 |
| `palette` | `str` | `"asii_3v"` | Имя палитры из `PALETTES` или своя строка символов (от тёмных к светлым) |
| `reverse_palette` | `bool` | `False` | Перевернуть палитру |
| `size` | `(w, h) \| None` | `None` | Целевой размер в символах; `None` — автоподбор под терминал |
| `fullcolor` | `bool` | `True` | `True` — ANSI truecolor, `False` — монохром |
| `color_levels` | `int \| None` | `None` | Уровней квантования на канал; `None` — вычисляется из размера палитры |
| `invert` | `bool` | `False` | Инвертировать яркость |
| `max_pixels` | `int \| None` | `32_000_000` | Даунскейл источника, если площадь больше лимита (защита RAM); `None` — без лимита |
| `edges` | `bool \| str \| None` | `False` | `True` → canny; либо `"canny"` / `"sobel"`; `False` — выключено |
| `edge_mode` | `str` | `"lines"` | `"lines"` — только контуры, `"overlay"` — контуры поверх оригинала |
| `low_threshold` / `high_threshold` | `int` | `50` / `150` | Пороги двойной фильтрации |
| `blur_ksize` | `int` | `5` | Гауссово размытие перед детекцией (`0` — выключить) |

### `convert_animation(path, *, ..., progress=False) -> Iterator[str]`

Те же параметры, что у `convert_image`, плюс `progress` (прогресс-бар). Возвращает **генератор**: каждый `next()` читает, обрабатывает и отдаёт ровно один ASCII-кадр, немедленно освобождая память. Длительность анимации не влияет на потребление RAM.

### `show_image(source, *, renderer=None, header="", save_path=None, **convert_kwargs) -> str`

`convert_image` + печать в консоль через `ConsoleRenderer`. Принимает все параметры конвертации из таблицы выше. Дополнительно:

* `renderer` — свой экземпляр `ConsoleRenderer` (например, с фиксированным `width/height`);
* `header` — строка-заголовок над артом;
* `save_path` — заодно сохранить результат в файл.

Возвращает сгенерированную строку.

### `play_animation(path, *, renderer=None, fps=None, duration=None, status_header=True, save_dir=None, **convert_kwargs) -> None`

Конвертация + пофреймовое проигрывание в терминале (стриминг, без хранения всех кадров):

* `fps` — переопределить частоту кадров источника;
* `duration` — ограничить время проигрывания, сек;
* `status_header` — служебная строка (имя файла, размер, кадр);
* `save_dir` — параллельно сохранять кадры в директорию (`.ans`).

### `save_ascii(text, path) -> str`

Сохраняет строку в файл (создаёт директории), возвращает путь.

### `terminal_fit_size(src_w, src_h, ...) -> (w, h)`

Автоподбор размера вывода под текущий терминал с сохранением пропорций (учитывает, что символ в ~2 раза выше, чем широк).

### Низкоуровневые конвертеры (`ascii_art_lib.converter`)

Работают с уже подготовленным кадром `ndarray` — используются внутри функций выше, но доступны и напрямую:

* `frame_to_symbols(frame, palette, size, *, reverse_palette=False, invert=False)` → `np.ndarray` символов;
* `symbols_to_text(sym)` → `str` (монохром);
* `frame_to_mono_text(frame, palette, size, *, ...)` → `str`;
* `frame_to_color_ansi(frame, palette, size, *, color_levels=None, ...)` → `str` с ANSI-кодами (дельта-кодирование: escape-последовательность печатается только при смене цвета);
* `quantize(values, levels)` — квантование массива яркости/цвета.

### Рендерер (`ascii_art_lib.renderer.ConsoleRenderer`)

Отдельный класс вывода в консоль:

```python
from ascii_art_lib import ConsoleRenderer, convert_animation

r = ConsoleRenderer()                 # авто-размер терминала; Windows: включён VT-режим
r.show_static(convert_image("a.png")) # статичный вывод
r.play(convert_animation("b.gif"), fps=24, duration=10, header_fn=lambda i, l: f"frame {i}")
r.save(text, "out.ans")               # сохранение кадра
r.clear()                             # очистка экрана (\033[2J\033[H])
```

Параметры конструктора: `width`, `height` (фиксация размера), `keep_aspect`, `clear_scrollback`, `out` (куда писать, по умолчанию `sys.stdout`). Метод `fit_size(src_w, src_h)` подбирает область вывода с сохранением пропорций.

---

## Палитры символов

Встроенные палитры (`ascii_art_lib.palettes.PALETTES`), упорядочены **от тёмных символов к светлым**:

| Имя | Длина | Авто-уровней на канал | Характер |
|---|---|---|---|
| `asii` | 68 | 18 | Классическая «плотная» палитра (Newell) |
| `asii_1` | 82 | 20 | Максимальная детализация |
| `asii_2` | 53 | 17 | Золотая середина |
| `asii_3` | 82 | 20 | Расширенная с пробелом в начале |
| `asii_3v` | 72 | 19 | **По умолчанию** — баланс деталей и читаемости |
| `asii_4` | 10 | 8 | Минималистичная (`" .;coPO?@#"`) |

Своя палитра — любая строка уникальных символов от тёмных к светлым:

```python
from ascii_art_lib import get_palette, PALETTES

convert_image("a.png", palette=" .:-=+*#%@")     # своя строка
convert_image("a.png", palette=get_palette("asii_4", reverse=True))  # реверс программно
```

Функция `get_palette(name_or_string, *, reverse=False)` резолвит имя или принимает пользовательскую строку; `reverse=True` разворачивает её. Тот же реверс доступен параметром `reverse_palette` во всех функциях и флагом `--reverse-palette` в CLI.

Бонус: словарь `SGA` — трансляция букв в символы «Alienese» (для пасхалок).

---

## Выделение контуров

Реализовано по мотивам [этого видео](https://youtu.be/gg40RWiaHRY): гауссово размытие → градиенты Собеля (Gx/Gy) → величина градиента → нормализация → двойной порог с гистерезисом (weak/strong + связность через dilation). Модуль: `ascii_art_lib.edges`.

### Через API-функции

```python
# чистый edge-art (только линии)
show_image("photo.png", edges=True)                      # canny по умолчанию
show_image("photo.png", edges="sobel", low_threshold=30)

# контуры поверх оригинального изображения
show_image("photo.png", edges="canny", edge_mode="overlay")

# в анимации — детектор создаётся один раз и переиспользуется по кадрам
play_animation("clip.mp4", edges=True, blur_ksize=3)
```

> 💡 Замечание из практики: контурный режим лучше всего смотрится в **монохроме на статичных изображениях** (`fullcolor=False, edges=True`); в цвете и анимации эффект может выглядеть шумно — поэтому он и не был включён по умолчанию.

### Отдельные функции и класс

```python
from ascii_art_lib import detect_edges, EdgeDetector, blend_with_source

# 1) карта контуров (uint8 0/255) — считается на уменьшенном кадре, если задан size
emap = detect_edges(cv2.imread("photo.png"), method="canny",
                    low_threshold=50, high_threshold=150, blur_ksize=5, size=(100, 50))

# 2) конфигурируемый класс для переиспользования (анимации, пресеты)
det = EdgeDetector(method="canny", low_threshold=40, high_threshold=120)
for frame in iter_frames("clip.mp4"):
    lines = det.apply(frame, mode="lines")      # чёрно-белые линии (готово к конвертации)
    mixed = det.apply(frame, mode="overlay")    # контуры поверх кадра
    text = convert_image(mixed, size=(100, 50))

# 3) ручное смешивание готовой карты с оригиналом
blended = blend_with_source(frame, emap, color=(0, 255, 255))
```

Все реализации векторизованы через C-функции OpenCV (`filter2D`, `magnitude`, `dilate`) — сложность `O(H·W)` без Python-циклов по пикселям.

### Палитра ориентации контуров (`edge_mode="palette"`)

У этого режима **своя собственная палитра**, которая выражается *наклоном линии*,
а не яркостью: символ выбирается по направлению касательной к контуру (угол
градиента Sobel, 8 секторов по 22.5°), поэтому ASCII-линия визуально сохраняет
свой наклон:

| Наклон линии | Символы |
|---|---|
| ~0°…45° (восходящая) | `/` |
| ~45°…90° (крутая/вертикальная) | `\|` |
| ~90°…135° (пологая/горизонтальная) | `-` |
| ~135°…180° (нисходящая) | `\\` |

Для **более сложных (изогнутых) контуров** расширенный набор добавляет парные
скобки: знак и направление кривизны оцениваются по Laplacian'у яркости,
нормированному на локальный контраст, и изгибы получают символ, «раскрывающийся»
в сторону вогнутости — `^ v < >` (горбы по сторонам света), `( ) [ ] { }`
(вертикальные/горизонтальные/диагональные дуги). Прямые участки при этом
остаются `/ - \ |`. Чувствительность переключения на скобки — `curve_threshold`
(меньше — больше изогнутых символов).

```python
from ascii_art_lib import convert_image, frame_to_edge_symbols, edge_symbols_to_text

# через высокоуровневый API (фон = цвет оригинала в цветном режиме)
art = convert_image("photo.png", edges=True, edge_mode="palette", fullcolor=False)
art_c = convert_image("photo.png", edges=True, edge_mode="palette", fullcolor=True)

# низкоуровнево: кадр -> uint8-массив байтов -> текст
sym = frame_to_edge_symbols(cv2.imread("photo.png"), size=(100, 50),
                            mode="extended", curve_threshold=0.5)
text = edge_symbols_to_text(sym)

# справочники палитры
from ascii_art_lib import EDGE_PALETTES, get_edge_palette, edge_palette_symbols
EDGE_PALETTES            # {'basic': '/|-\\', 'extended': '/|-\\^v<>()[]{}'}
get_edge_palette("basic")
```

CLI: `--edge-mode palette` плюс `--curve-threshold X`. В сочетании с `--no-color`
получается чистый «рисовальщик линий», с цветом — линии красятся truecolor-цветом
оригинала в этих точках. Модуль: `ascii_art_lib.edge_palette` (весь расчёт —
табличные векторные операции NumPy/OpenCV, O(H·W)).

---

## Цветовой охват и палитра

Цветовое разрешение вывода **автоматически зависит от выбранной палитры** — логика в `palette_color_levels(palette)`:

```
levels = round(2 + 2·√len(palette)), ограничено диапазоном [2 .. 64]
```

| Палитра | Длина | Авто-уровней на канал |
|---|---|---|
| `asii_4` | 10 | 8 |
| `asii_3v` | 72 | 19 |
| `asii_1` | 82 | 20 |

Смысл: бедная палитра всё равно не различит мелкие цветовые градации — избыточное квантование только раздувает ANSI-выход; богатая палитра получает соответствующий охват. Если нужно задать охват жёстко — явно передайте `color_levels`, и зависимость от палитры переопределяется:

```python
convert_image("a.png", palette="asii_4", color_levels=64)  # максимум цвета при минимуме символов
convert_image("a.png", fullcolor=False)                     # цвета нет вовсе
```

В CLI — флаг `--color-levels N`.

---

## Оптимизация памяти и производительности

**Память:**

1. **Стриминг анимаций.** `iter_frames()` читает GIF/видео покадрово; `convert_animation` — генератор, который отдаёт ASCII-кадр и делает `del frame` до чтения следующего. Потребление RAM ≈ один исходный кадр + один текстовый кадр, **не зависит от числа кадров** (в исходной версии все кадры загружались в список).
2. **Лимит `max_pixels`.** Гигантские файлы (например, 100×100 Мп) предварительно даунскейлятся (`INTER_AREA`) до разумной площади — защита от OOM; управляется параметром/флагом `--max-pixels`.
3. **Кадры контурной обработки считаются уже на уменьшенном до целевого размера изображении** (параметр `size` в `detect_edges`) — промежуточных буферов в разы меньше.
4. Немедленное освобождение промежуточных массивов (`del gx, gy` и т.п.) в горячих циклах.

**Производительность:**

1. **Полная векторизация NumPy**: выбор символов через LUT-таблицу / `np.take` вместо `np.vectorize`-обёрток над Python-функциями; квантование цвета — целочисленная арифметика над всем массивом сразу.
2. **Дельта-кодирование ANSI**: цветовая escape-последовательность вставляется только при *смене* цвета соседних символов — output строится склейкой частей, поток вывода заметно короче.
3. **OpenCV C-ядра** (`resize INTER_AREA`, `GaussianBlur`, `filter2D`, `magnitude`, `dilate`) вместо свёрток на чистом Python.
4. **Обработка на целевом размере**: ресайз выполняется до конвертации, все дальнейшие операции идут над массивом порядка `W_терминала × H_терминала`, а не исходника.
5. LUT палитр кэшируются (`functools.lru_cache`) — при анимации таблица строится один раз.

---

## Структура проекта

```
.
├── ascii_art_lib/            # пакет
│   ├── __init__.py           # публичные экспорты (__all__)
│   ├── __main__.py           # python -m ascii_art_lib
│   ├── api.py                # convert_image / convert_animation / show_image /
│   │                         # play_animation / save_ascii / terminal_fit_size
│   ├── cli.py                # argparse-интерфейс (asciiart)
│   ├── converter.py          # низкоуровневые векторные конвертеры (LUT, ANSI)
│   ├── edges.py              # detect_edges / EdgeDetector / blend_with_source
│   ├── media.py              # classify / probe / iter_frames (стриминг кадров)
│   ├── palettes.py           # палитры, get_palette(reverse), palette_color_levels, LUT
│   ├── renderer.py           # ConsoleRenderer — вывод статики и анимации в консоль
│   └── threshold_map.py      # исторический модуль карт порогов (Sobel-ядра)
├── main.py                   # запуск без установки: python main.py <файл> [...]
├── tests/test_ascii_art.py   # pytest-тесты
├── pyproject.toml            # метаданные пакета + entry point `asciiart`
└── requirements.txt
```

Слои: `cli/api` → `converter/edges/renderer` → `media/palettes`. Пользовательский код обычно трогает только верхний слой.

---

## Тесты

```bash
pip install -e .[dev]
pytest tests/ -v
```

Покрыто: загрузка/классификация медиа, палитры и реверс, квантование, монохромный и ANSI-вывод, детекция контуров (canny/sobel, EdgeDetector), комбинации `edges + color + invert`, зависимость цветового охвата от палитры, флаги CLI.

---

## Лицензия

MIT.