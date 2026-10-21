#region Импорты и типы
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image, ImageDraw, ImageFont

from scipy import ndimage
from scipy.spatial import ConvexHull

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# Типы данных
CharData = Dict[str, Any]
ClusterGroup = Dict[int, List[CharData]]
MetricsDict = Dict[str, float]
PaletteEntry = Dict[str, Any]
#endregion Импорты и типы

#region Параметры и константы
OUTPUT_DIR: str = "output" 

CELL_W: int = 40 # 20
CELL_H: int = 60 # 30
FONT_SIZE: int = 24
FONT_PATH: str = "consola.ttf"
THRESHOLD: int = 128
K_RANGE: range = range(3, 12)

USE_MANUAL_K: bool = False   # True — задать k вручную, False — автопоиск
MANUAL_K: int = 7           # желаемое число кластеров

# TODO: Пресеты фильтрации
# PRESET_LETTERS_ONLY = {
#     "EXCLUDE_CHARS": [],
#     "EXCLUDE_CATEGORIES": ["whitespace", "punctuation", "digits"],
#     "INCLUDE_ONLY": []
# }

# PRESET_NO_PUNCTUATION = {
#     "EXCLUDE_CHARS": [" "],
#     "EXCLUDE_CATEGORIES": ["punctuation"],
#     "INCLUDE_ONLY": []
# }

# PRESET_CUSTOM = {
#     "EXCLUDE_CHARS": [],
#     "EXCLUDE_CATEGORIES": [],
#     "INCLUDE_ONLY": ["@", "#", "$", "M", "W", "B", "0", "8"]
# }

# # Активный пресет
# ACTIVE_PRESET = PRESET_NO_PUNCTUATION

# Фильтрация символов
EXCLUDE_CHARS: List[str] = [" "]  # Конкретные символы для исключения
EXCLUDE_CATEGORIES: List[str] = []  # Категории: "whitespace", "control", "punctuation", "digits", "lowercase", "uppercase"
INCLUDE_ONLY: List[str] = []  # Если не пустой, анализировать только эти символы (игнорирует EXCLUDE)


# Метрики, по которым кластеризуем
#
# Old
# CLUSTER_FEATURES: List[str] = [
#     "brightness", "density",
#     "aspect", "cx", "cy",
#     "solidity", "extent", "circularity",
#     "holes", "components", "euler_number",
#     "h_var", "v_var", "block_var",
#     "edge_density", "h_sym", "v_sym",
# ]
#
#New
CLUSTER_FEATURES: List[str] = [
    "euler_number",    # Топология
    "holes",           # Количество дырок
    "block_var",       # Равномерность
    "edge_density",    # Детализированность
    "extent",          # Заполненность bbox
    "h_var", "v_var",  # Направленность
    "h_sym", "v_sym",  # Симметрия
]

#endregion Параметры и константы

#region Рендер и метрики

def render_char(ch: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """Рендер символа в бинарную маску."""
    img: Image.Image = Image.new("L", (CELL_W, CELL_H), 255)
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(img)
    bbox: Tuple[int, int, int, int] = draw.textbbox((0, 0), ch, font=font)
    x: int = (CELL_W - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y: int = (CELL_H - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), ch, fill=0, font=font)
    mask: np.ndarray = np.array(img) < THRESHOLD
    return mask

def get_char_category(ch: str) -> str:
    """Определяет категорию символа."""
    if ch.isspace():
        return "whitespace"
    elif ch.isalpha() and ch.islower():
        return "lowercase"
    elif ch.isalpha() and ch.isupper():
        return "uppercase"
    elif ch.isdigit():
        return "digits"
    elif ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
        return "punctuation"
    else:
        return "other"


def should_include_char(ch: str) -> bool:
    """Проверяет, должен ли символ быть включён в анализ."""
    # Если INCLUDE_ONLY не пуст, проверяем только его
    if INCLUDE_ONLY:
        return ch in INCLUDE_ONLY
    
    # Проверяем конкретные исключения
    if ch in EXCLUDE_CHARS:
        return False
    
    # Проверяем категории
    category = get_char_category(ch)
    if category in EXCLUDE_CATEGORIES:
        return False
    
    return True

def compute_metrics(mask: np.ndarray) -> MetricsDict:
    """Вычисляет все метрики для бинарной маски."""
    filled: int = int(mask.sum())
    total: int = mask.size
    
    if filled == 0:
        zero_metrics: MetricsDict = {k: 0.0 for k in CLUSTER_FEATURES + ["brightness", "density"]}
        return zero_metrics

    # Bounding box
    rows: np.ndarray = np.any(mask, axis=1)
    cols: np.ndarray = np.any(mask, axis=0)
    row_indices: np.ndarray = np.where(rows)[0]
    col_indices: np.ndarray = np.where(cols)[0]
    rmin: int = int(row_indices[0])
    rmax: int = int(row_indices[-1])
    cmin: int = int(col_indices[0])
    cmax: int = int(col_indices[-1])
    bbox_w: int = cmax - cmin + 1
    bbox_h: int = rmax - rmin + 1
    aspect: float = bbox_w / bbox_h if bbox_h else 0.0

    # Центр масс
    ys: np.ndarray
    xs: np.ndarray
    ys, xs = np.where(mask)
    cx: float = float(xs.mean()) / CELL_W
    cy: float = float(ys.mean()) / CELL_H

    # Компактность
    kernel: np.ndarray = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    edges: np.ndarray = ndimage.convolve(mask.astype(int), kernel) != 0
    
    # perimeter: int = int(edges.sum())
    # compactness: float = (perimeter ** 2) / filled

    # Топология - ИСПРАВЛЕНО
    labeled: np.ndarray
    n_components: int
    labeled, n_components = ndimage.label(mask)  # Убран return_num=True
    bbox_mask: np.ndarray = np.zeros_like(mask)
    bbox_mask[rmin:rmax+1, cmin:cmax+1] = True
    bg: np.ndarray = (~mask) & bbox_mask
    holes: int = 0
    
    solidity: float = compute_solidity(mask)
    extent: float = compute_extent(mask)
    circularity: float = compute_circularity(mask)
    euler_number: float = float(compute_euler_number(mask))
    
    if bg.any():
        bg_labeled: np.ndarray
        n_bg: int
        bg_labeled, n_bg = ndimage.label(bg)  # Убран return_num=True
        for lbl in range(1, n_bg + 1):
            comp: np.ndarray = bg_labeled == lbl
            if not (comp[0].any() or comp[-1].any() or
                    comp[:, 0].any() or comp[:, -1].any()):
                holes += 1

    # Проекционные дисперсии
    h_proj: np.ndarray = mask.sum(axis=1).astype(float)
    v_proj: np.ndarray = mask.sum(axis=0).astype(float)
    h_var: float = float(h_proj.var()) / (filled ** 2)
    v_var: float = float(v_proj.var()) / (filled ** 2)

    # Block variance
    mid_y: int = CELL_H // 2
    mid_x: int = CELL_W // 2
    blocks: List[float] = [
        float(mask[:mid_y, :mid_x].sum()),
        float(mask[:mid_y, mid_x:].sum()),
        float(mask[mid_y:, :mid_x].sum()),
        float(mask[mid_y:, mid_x:].sum()),
    ]
    block_var: float = float(np.var(blocks)) / (filled ** 2)

    # Edge density
    edge_density: float = float(edges.sum()) / filled
    
    # Симметрии
    h_sym: float = float(np.minimum(mask, mask[:, ::-1]).sum()) / filled
    v_sym: float = float(np.minimum(mask, mask[::-1, :]).sum()) / filled

    metrics: MetricsDict = {
        "brightness":   float(filled),
        "density":      filled / total,
        "aspect":       aspect,
        "cx":           cx,
        "cy":           cy,
        "solidity":     solidity, 
        "extent":       extent, 
        "circularity":  circularity, 
        "euler_number": euler_number,
        "holes":        float(holes),
        "components":   float(n_components),
        "h_var":        h_var,
        "v_var":        v_var,
        "block_var":    block_var,
        "edge_density": edge_density,
        "h_sym":        h_sym,
        "v_sym":        v_sym,
    }
    return metrics

def compute_solidity(mask: np.ndarray) -> float:
    """Вычисляет solidity (отношение площади к площади выпуклой оболочки)."""
    filled = int(mask.sum())
    if filled == 0:
        zero_metrics: MetricsDict = {
        k: 0.0 for k in CLUSTER_FEATURES + ["brightness", "density",
                                             "solidity", "extent",
                                             "circularity", "euler_number"]
        }
        return zero_metrics
    
    # Находим координаты заполненных пикселей
    ys, xs = np.where(mask)
    points = np.column_stack([xs, ys])
    
    if len(points) < 3:
        return 1.0  # Точка или линия — считаем выпуклой
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # В 2D volume = area
        return filled / hull_area if hull_area > 0 else 1.0
    except Exception:
        return 1.0

def compute_extent(mask: np.ndarray) -> float:
    """Вычисляет extent (отношение площади к площади bounding box)."""
    filled = int(mask.sum())
    if filled == 0:
        return 0.0
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
    return filled / bbox_area if bbox_area > 0 else 0.0

def compute_circularity(mask: np.ndarray) -> float:
    """Вычисляет circularity (насколько символ близок к кругу)."""
    filled = int(mask.sum())
    if filled == 0:
        return 0.0
    
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    edges = ndimage.convolve(mask.astype(int), kernel) != 0
    perimeter = int(edges.sum())
    
    if perimeter == 0:
        return 0.0
    
    return 4 * np.pi * filled / (perimeter ** 2)

def compute_euler_number(mask: np.ndarray) -> int:
    """Вычисляет Euler number (компоненты - дырки)."""
    labeled, n_components = ndimage.label(mask)
    
    # Считаем дырки (как в compute_metrics)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    bbox_mask = np.zeros_like(mask)
    bbox_mask[rmin:rmax+1, cmin:cmax+1] = True
    bg = (~mask) & bbox_mask
    
    holes = 0
    if bg.any():
        bg_labeled, n_bg = ndimage.label(bg)
        for lbl in range(1, n_bg + 1):
            comp = bg_labeled == lbl
            if not (comp[0].any() or comp[-1].any() or
                    comp[:, 0].any() or comp[:, -1].any()):
                holes += 1
    
    return n_components - holes

def normalize_metrics(data: List[CharData], to_percent: bool = False) -> List[CharData]:
    """
    Нормализует все числовые метрики к [0, 1] или [0, 100].
    
    Args:
        data: список словарей с метриками
        to_percent: если True, умножает на 100 (проценты)
    
    Returns:
        нормализованные данные
    """
    # Список метрик для нормализации (без category)
    metrics_to_normalize = [
        "brightness", "density",
        "aspect", "cx", "cy",
        "solidity", "extent", "circularity",
        "holes", "components", "euler_number",
        "h_var", "v_var", "block_var",
        "edge_density", "h_sym", "v_sym",
    ]
    
    # Матрица признаков
    X = np.array([[c[m] for m in metrics_to_normalize] for c in data])
    
    # MinMaxScaler с обрезкой выбросов
    scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    X_normalized = scaler.fit_transform(X)
    
    # Записываем обратно
    for i, c in enumerate(data):
        for j, m in enumerate(metrics_to_normalize):
            value = float(X_normalized[i, j])
            if to_percent:
                value *= 100
            c[m] = round(value, 4)
    
    return data

#endregion Рендер и метрики

#region Анализ символов
def analyze() -> List[CharData]:
    """Анализ ASCII-символов с фильтрацией."""
    try:
        font: ImageFont.FreeTypeFont = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    results: List[CharData] = []
    excluded_count: int = 0
    
    for code in range(32, 127):
        ch: str = chr(code)
        
        # Фильтрация
        if not should_include_char(ch):
            excluded_count += 1
            continue
        
        mask: np.ndarray = render_char(ch, font)
        m: MetricsDict = compute_metrics(mask)
        m["char"] = ch
        m["ord"] = code
        m["category"] = get_char_category(ch)  # Добавляем категорию
        results.append(m)
    
    print(f"[i] Проанализировано: {len(results)} символов, исключено: {excluded_count}")
    return results
#endregion Анализ символов

#region Кластеризация
def find_best_k(X: np.ndarray) -> Tuple[int, Dict[int, float]]:
    """Подбирает k по silhouette score с elbow method."""
    best_k: int = 2
    best_score: float = -1.0
    scores: Dict[int, float] = {}
    inertias: Dict[int, float] = {}
    
    for k in K_RANGE:
        km: KMeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels: np.ndarray = km.fit_predict(X)
        score: float = silhouette_score(X, labels)
        scores[k] = score
        inertias[k] = float(km.inertia_)
        
        # Elbow: ищем максимальное падение inertia
        # Проверяем, что оба предыдущих ключа существуют
        if (k - 1) in inertias and (k - 2) in inertias:
            drop: float = inertias[k - 1] - inertias[k]
            prev_drop: float = inertias[k - 2] - inertias[k - 1]
            if prev_drop > 0 and drop < prev_drop * 0.5 and best_k == 2:
                best_k = k - 1
    
    # Если elbow не сработал, используем silhouette score
    if best_k == 2:
        for k, score in scores.items():
            if score > best_score:
                best_score = score
                best_k = k
    
    return best_k, scores



def cluster_chars(data: List[CharData]) -> Tuple[
    ClusterGroup,
    np.ndarray,
    np.ndarray,
    List[CharData],
    Dict[int, float],
    int
]:
    """Кластеризует символы и возвращает группы."""
    # Исключаем пробел
    #space: CharData = next(d for d in data if d["char"] == " ")
    chars: List[CharData] = [d for d in data if d["char"] != " "]

    # Матрица признаков
    X: np.ndarray = np.array([[d[f] for f in CLUSTER_FEATURES] for d in chars])
    scaler: StandardScaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(X)

    best_k: int
    scores: Dict[int, float]
    if USE_MANUAL_K:
        best_k: int = MANUAL_K
        scores: Dict[int, float] = {}
        print(f"[i] Используем ручной k={best_k}")
    else:
        best_k, scores = find_best_k(X_scaled)
        print(f"[i] Оптимальное число кластеров: {best_k} "
            f"(silhouette = {scores[best_k]:.3f})")

    km: KMeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels: np.ndarray = km.fit_predict(X_scaled)

    # PCA для визуализации
    pca: PCA = PCA(n_components=2)
    X_2d: np.ndarray = pca.fit_transform(X_scaled)

    # Группируем по кластерам
    groups: ClusterGroup = {i: [] for i in range(best_k)}
    for d, lbl, xy in zip(chars, labels, X_2d):
        char_with_pca: CharData = {**d, "pca_x": float(xy[0]), "pca_y": float(xy[1])}
        groups[lbl].append(char_with_pca)

    # Сортируем внутри каждого кластера по brightness
    for g in groups.values():
        g.sort(key=lambda c: c["brightness"])

    return groups, X_2d, labels, chars, scores, best_k
#endregion Кластеризация

#region Описание кластеров

def describe_cluster(cluster_chars: List[CharData]) -> str:
    avg: MetricsDict = {f: float(np.mean([c[f] for c in cluster_chars])) 
                        for f in CLUSTER_FEATURES}

    tags: List[str] = []

    # Яркость и плотность (теперь в [0, 1])
    if avg["brightness"] > 0.7:
        tags.append("тёмные")
    elif avg["brightness"] < 0.3:
        tags.append("светлые")
    
    if avg["density"] > 0.6:
        tags.append("плотные")
    elif avg["density"] < 0.2:
        tags.append("разреженные")

    # Топология
    if avg["holes"] > 0.5:
        tags.append("дырчатые")
    if avg["components"] > 0.3:
        tags.append("разрозненные")
    if avg["holes"] < 0.1 and avg["components"] < 0.2:
        tags.append("сплошные")

    # Euler number (нормализован, может быть отрицательным)
    if avg["euler_number"] < 0.3:
        tags.append("топологически сложные")
    elif avg["euler_number"] > 0.7:
        tags.append("многокомпонентные")

    # Форма
    if avg["solidity"] > 0.8:
        tags.append("выпуклые")
    elif avg["solidity"] < 0.5:
        tags.append("вогнутые")

    if avg["extent"] > 0.7:
        tags.append("плотно заполняющие ячейку")
    elif avg["extent"] < 0.3:
        tags.append("малые относительно ячейки")

    if avg["circularity"] > 0.7:
        tags.append("округлые")
    elif avg["circularity"] < 0.3:
        tags.append("угловатые/линейные")

    # Направленность
    if avg["h_var"] > avg["v_var"] * 1.5 and avg["h_var"] > 0.1:
        tags.append("горизонтальные")
    elif avg["v_var"] > avg["h_var"] * 1.5 and avg["v_var"] > 0.1:
        tags.append("вертикальные")

    # Симметрия
    if avg["h_sym"] > 0.8 and avg["v_sym"] > 0.8:
        tags.append("симметричные")
    elif avg["h_sym"] < 0.4 and avg["v_sym"] < 0.4:
        tags.append("асимметричные")

    # Равномерность
    if avg["block_var"] < 0.1:
        tags.append("равномерные")
    elif avg["block_var"] > 0.3:
        tags.append("угловатые")

    # Edge density
    if avg["edge_density"] > 0.7:
        tags.append("резкие")
    elif avg["edge_density"] < 0.3:
        tags.append("мягкие")

    return ", ".join(tags) if tags else "смешанные"

#endregion Описание кластеров

#region Визуализация
def plot_clusters(
    #groups: ClusterGroup,
    X_2d: np.ndarray,
    labels: np.ndarray,
    chars: List[CharData],
    scores: Dict[int, float],
    best_k: int
) -> None:
    """Визуализация кластеров."""
    fig: plt.Figure
    axes: np.ndarray
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Scatter с PCA
    ax: plt.Axes = axes[0]
    cmap = plt.get_cmap("tab10")
    for i in range(best_k):
        mask: np.ndarray = labels == i
        pts: np.ndarray = X_2d[mask]
        chs: List[str] = [chars[j]["char"] for j in np.where(mask)[0]]
        ax.scatter(pts[:, 0], pts[:, 1], c=[cmap(i)], s=80,
                   label=f"Cluster {i}", edgecolor="black")
        for p, ch in zip(pts, chs):
            ax.text(p[0], p[1], ch, fontsize=8, ha="center", va="center",
                    color="white", fontweight="bold")
    ax.set_title(f"ASCII clusters (PCA, k={best_k}, "
                 f"silhouette={scores[best_k]:.3f})")
    ax.set_xlabel(f"PC1")
    ax.set_ylabel(f"PC2")
    ax.legend()
    ax.grid(alpha=0.3)

    # Silhouette curve
    ax = axes[1]
    ax.plot(list(scores.keys()), list(scores.values()), "o-")
    ax.axvline(best_k, color="red", linestyle="--", label=f"best k={best_k}")
    ax.set_xlabel("k (число кластеров)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Выбор оптимального k")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ascii_clusters.png", dpi=150)
    plt.show()
    
def plot_brightness_vs_metric(
    data: List[CharData],
    metric: str = "edge_density",
    metric_label: str = "Edge Density"
) -> None:
    """График: яркость vs выбранная метрика."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Исключаем пробел
    chars: List[CharData] = [c for c in data if c["char"] != " "]
    brightness: List[float] = [c["brightness"] for c in chars]
    metric_values: List[float] = [c[metric] for c in chars]
    char_labels: List[str] = [c["char"] for c in chars]
    
    # Scatter с цветовой кодировкой по яркости
    scatter = ax.scatter(brightness, metric_values, c=brightness, 
                         cmap="gray_r", s=100, edgecolors="black", 
                         linewidths=0.5, alpha=0.8, zorder=2)
    
    # Подписи символов
    for x, y, ch in zip(brightness, metric_values, char_labels):
        # Белый текст для тёмных символов (высокая яркость), чёрный для светлых
        text_color: str = "white" if x > 150 else "black"
        ax.text(x, y, ch, fontsize=10, ha="center", va="center",
                color=text_color, fontweight="bold", zorder=3)
    
    ax.set_xlabel("Brightness (заполненные пиксели)", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"ASCII символы: яркость vs {metric_label}", fontsize=14)
    ax.grid(alpha=0.3, zorder=1)
    
    # Цветовая шкала
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Brightness", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ascii_brightness_vs_{metric}.png", dpi=150)
    plt.show()
    
#endregion Визуализация

#region Основная функция
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data: List[CharData] = analyze()
    
    data = normalize_metrics(data, to_percent=False)  # или True для процентов
    

    # Сохраняем полные метрики
    with open(f"{OUTPUT_DIR}/ascii_full.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    groups: ClusterGroup
    X_2d: np.ndarray
    labels: np.ndarray
    chars: List[CharData]
    scores: Dict[int, float]
    best_k: int
    groups, X_2d, labels, chars, scores, best_k = cluster_chars(data)

    # Для PCA-подписей
    pca: PCA = PCA(n_components=2)
    X_features: np.ndarray = np.array([[d[f] for f in CLUSTER_FEATURES] for d in chars])
    scaler: StandardScaler = StandardScaler()
    pca.fit(scaler.fit_transform(X_features))
    pca_explained: np.ndarray = pca.explained_variance_ratio_

    # Вывод классов
    print("\n" + "=" * 70)
    print("СЕМАНТИЧЕСКАЯ ПАЛИТРА ASCII-СИМВОЛОВ")
    print("=" * 70)

    # Сортируем классы по средней яркости
    sorted_groups: List[Tuple[int, List[CharData]]] = sorted(
        groups.items(),
        key=lambda kv: float(np.mean([c["brightness"] for c in kv[1]]))
    )

    for rank, (cid, members) in enumerate(sorted_groups, 1):
        #desc: str = describe_cluster(members)
        avg_b: float = float(np.mean([c["brightness"] for c in members]))
        avg_d: float = float(np.mean([c["density"] for c in members]))
        chars_str: str = "".join(c["char"] for c in members)

        # print(f"\n▶ КЛАСС {rank}  |  {desc}")
        print(f"\n▶ КЛАСС {rank}")
        print(f"  размер: {len(members):>2}  |  "
              f"яркость: {avg_b:>6.1f}  |  плотность: {avg_d:.3f}")
        print(f"  символы: {chars_str}")

    #plot_clusters(groups, X_2d, labels, chars, scores, best_k)
    plot_clusters(X_2d, labels, chars, scores, best_k)
    
    # График яркость vs edge density
    # plot_brightness_vs_metric(data, "edge_density", "Edge Density (детализированность)")

    # # Альтернативные варианты (раскомментируй при необходимости):
    # plot_brightness_vs_metric(data, "holes", "Количество дырок")
    # plot_brightness_vs_metric(data, "block_var", "Block Variance (равномерность)")
    # plot_brightness_vs_metric(data, "solidity", "Solidity (выпуклость)")
    # plot_brightness_vs_metric(data, "extent", "Extent (заполненность bbox)")
    # plot_brightness_vs_metric(data, "circularity", "Circularity (округлость)")
    # plot_brightness_vs_metric(data, "euler_number", "Euler Number (топология)")
    

    # Сохраняем палитру
    palette: List[PaletteEntry] = []
    for rank, (cid, members) in enumerate(sorted_groups, 1):
        entry: PaletteEntry = {
            "rank": rank,
            #"description": describe_cluster(members),
            "chars": [c["char"] for c in members],
            "avg_brightness": float(np.mean([c["brightness"] for c in members])),
            "avg_density":    float(np.mean([c["density"] for c in members])),
        }
        palette.append(entry)
    
    with open(f"{OUTPUT_DIR}/ascii_palette.json", "w", encoding="utf-8") as f:
        json.dump(palette, f, ensure_ascii=False, indent=2)
    print(f"\n[i] Палитра сохранена в ascii_palette.json")
#endregion Основная функция