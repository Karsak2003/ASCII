#region Импорты и типы
import os
import json
from pickle import TRUE
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

from compute_metrics import (
    CELL_W, CELL_H,
    EXCLUDE_CHARS, EXCLUDE_CATEGORIES, INCLUDE_ONLY,
    CLUSTER_FEATURES, compute_metrics,
    CharMetrics
)

# Типы данных
ClusterGroup = Dict[int, List[CharMetrics]]
MetricsDict = Dict[str, float]
PaletteEntry = Dict[str, Any]
#endregion Импорты и типы

#region Параметры и константы
OUTPUT_DIR: str = "output" 

CELL_W: int = 50 # 20
CELL_H: int = 100 # 30


FONT_SIZE: int = 24
FONT_PATH: str = "consola.ttf"
THRESHOLD: int = 128
K_RANGE: range = range(3, 12)

USE_MANUAL_K: bool = True   # True — задать k вручную, False — автопоиск
MANUAL_K: int = 8           # желаемое число кластеров

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

def normalize_metrics(data: List[CharMetrics], to_percent: bool = False) -> List[CharMetrics]:
    """Нормализует только актуальные метрики."""
    metrics_to_normalize = CLUSTER_FEATURES + ["brightness", "density"]
    
    X = np.array([[getattr(c, m) for m in metrics_to_normalize] for c in data])
    scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    X_normalized = scaler.fit_transform(X)
    
    for i, c in enumerate(data):
        for j, m in enumerate(metrics_to_normalize):
            value = float(X_normalized[i, j])
            if to_percent:
                value *= 100
            setattr(c, m, round(value, 4))
    return data
#endregion Рендер и метрики

#region Анализ символов
def analyze() -> List[CharMetrics]:
    """Анализ ASCII-символов с фильтрацией."""
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    results: List[CharMetrics] = []
    excluded_count = 0
    
    for code in range(32, 127):
        ch = chr(code)
        if not should_include_char(ch):
            excluded_count += 1
            continue
        
        mask = render_char(ch, font)
        category = get_char_category(ch)
        metrics = compute_metrics(mask, ch, code, category)
        results.append(metrics)

    print(f"[i] Проанализировано: {len(results)} символов, исключено: {excluded_count} ")
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

def cluster_chars(data: List[CharMetrics]) -> Tuple[Dict[int, List[CharMetrics]], np.ndarray, np.ndarray, List[CharMetrics], Dict[int, float], int]:
    """Кластеризует символы и возвращает группы."""
    chars = [c for c in data if c.char != " "]
    
    # Матрица признаков через getattr для объектов dataclass
    X = np.array([[getattr(c, f) for f in CLUSTER_FEATURES] for c in chars])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if USE_MANUAL_K:
        best_k = MANUAL_K
        _, scores = find_best_k(X_scaled)
        print(f"[i] Используем ручной k={best_k} ",
            f"(silhouette при этом k = {scores.get(best_k, 0):.3f})")
        print(f"[i] Для сравнения: лучший по silhouette был бы k={max(scores, key=scores.get)} ",
            f"со score={max(scores.values()):.3f}")
        
    else:
        best_k, scores = find_best_k(X_scaled)
        print(f"[i] Оптимальное число кластеров: {best_k} (silhouette = {scores[best_k]:.3f})")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    groups: Dict[int, List[CharMetrics]] = {i: [] for i in range(best_k)}
    for c, lbl, xy in zip(chars, labels, X_2d):
        c.pca_x = float(xy[0])  # Добавляем динамически для визуализации
        c.pca_y = float(xy[1])
        groups[lbl].append(c)

    for g in groups.values():
        g.sort(key=lambda item: item.brightness)

    return groups, X_2d, labels, chars, scores, best_k

#endregion Кластеризация

#region Визуализация

def plot_clusters(
    X_2d: np.ndarray,
    labels: np.ndarray,
    chars: List[CharMetrics],
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
        # ИСПРАВЛЕНО: c.char вместо c["char"]
        chs: List[str] = [chars[j].char for j in np.where(mask)[0]]
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
    data: List[CharMetrics],
    metric: str = "edge_density",
    metric_label: str = "Edge Density"
) -> None:
    """График: яркость vs выбранная метрика."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Исключаем пробел (ИСПРАВЛЕНО: c.char)
    chars: List[CharMetrics] = [c for c in data if c.char != " "]
    
    # ИСПРАВЛЕНО: обращение через точку и getattr для динамической метрики
    brightness: List[float] = [c.brightness for c in chars]
    metric_values: List[float] = [getattr(c, metric, 0.0) for c in chars]
    char_labels: List[str] = [c.char for c in chars]
    
    # Scatter с цветовой кодировкой по яркости
    scatter = ax.scatter(brightness, metric_values, c=brightness, 
                         cmap="gray_r", s=100, edgecolors="black", 
                         linewidths=0.5, alpha=0.8, zorder=2)
    
    # Подписи символов
    for x, y, ch in zip(brightness, metric_values, char_labels):
        # Белый текст для тёмных символов (высокая яркость), чёрный для светлых
        text_color: str = "white" if x > 0.5 else "black" # ИСПРАВЛЕНО: порог 0.5 т.к. данные нормализованы [0, 1]
        ax.text(x, y, ch, fontsize=10, ha="center", va="center",
                color=text_color, fontweight="bold", zorder=3)
    
    ax.set_xlabel("Brightness (нормализованная)", fontsize=12)
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
    
    data = analyze()
    data = normalize_metrics(data, to_percent=False)
    
    # Сохраняем полные метрики (используем to_dict() из dataclass)
    with open(f"{OUTPUT_DIR}/ascii_full.json", "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in data], f, ensure_ascii=False, indent=2)

    groups, X_2d, labels, chars, scores, best_k = cluster_chars(data)

    print("\n" + "=" * 70)
    print("СЕМАНТИЧЕСКАЯ ПАЛИТРА ASCII-СИМВОЛОВ")
    print("=" * 70)

    # Сортируем классы по средней яркости (ИСПРАВЛЕНО: c.brightness)
    sorted_groups: List[Tuple[int, List[CharMetrics]]] = sorted(
        groups.items(),
        key=lambda kv: float(np.mean([c.brightness for c in kv[1]]))
    )

    palette: List[PaletteEntry] = []
    for rank, (cid, members) in enumerate(sorted_groups, 1):
        avg_b: float = float(np.mean([c.brightness for c in members]))
        avg_d: float = float(np.mean([c.density for c in members]))
        chars_str: str = "".join(c.char for c in members)
        # Можно раскомментировать, если добавишь метод describe() в CharMetrics
        desc: str = members[0].describe() 
        
        print(f"\n▶ КЛАСС {rank} | {desc}")
        print(f"  размер: {len(members):>2}  |  "
              f"яркость: {avg_b:>6.1f}  |  плотность: {avg_d:.3f}")
        print(f"  символы: {chars_str}")

        palette.append({
            "rank": rank,
            # "description": desc,
            "chars": [c.char for c in members],
            "avg_brightness": avg_b,
            "avg_density": avg_d,
        })

    plot_clusters(X_2d, labels, chars, scores, best_k)
    
    # Раскомментируй для построения графиков:
    # plot_brightness_vs_metric(data, "linearity", "Linearity (линеобразность)")
    # plot_brightness_vs_metric(data, "holes", "Holes (дырки)")

    with open(f"{OUTPUT_DIR}/ascii_palette.json", "w", encoding="utf-8") as f:
        json.dump(palette, f, ensure_ascii=False, indent=2)
    print(f"\n[i] Палитра сохранена в {OUTPUT_DIR}/ascii_palette.json")
    
    
    
#endregion Основная функция