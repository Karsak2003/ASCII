#region Импорты и типы

import numpy as np
from typing import Dict, List, Any


from scipy import ndimage
from scipy.spatial import ConvexHull

from sklearn.decomposition import PCA

from skimage.morphology import skeletonize

from symbol_parser.Char_Metrics import CharMetrics

# Типы данных
CharData = Dict[str, Any]
ClusterGroup = Dict[int, List[CharData]]
MetricsDict = Dict[str, float]
PaletteEntry = Dict[str, Any]
#endregion Импорты и типы

#region Параметры и константы

CELL_W: int = 40 # 20
CELL_H: int = 60 # 30

PRESET_LETTERS_ONLY = {
    "EXCLUDE_CHARS": [],
    "EXCLUDE_CATEGORIES": ["whitespace", "punctuation", "digits"],
    "INCLUDE_ONLY": []
}

PRESET_NO_PUNCTUATION = {
    "EXCLUDE_CHARS": [" "],
    "EXCLUDE_CATEGORIES": ["punctuation"],
    "INCLUDE_ONLY": []
}

PRESET_CUSTOM = {
    "EXCLUDE_CHARS": [],
    "EXCLUDE_CATEGORIES": [],
    "INCLUDE_ONLY": ["@", "#", "$", "M", "W", "B", "0", "8"]
}

PRESET_NULL = {
    "EXCLUDE_CHARS": [" "],
    "EXCLUDE_CATEGORIES": [],
    "INCLUDE_ONLY": []
}

# Активный пресет
ACTIVE_PRESET = PRESET_NULL

# # Фильтрация символов
# EXCLUDE_CHARS: List[str] = [" "]  # Конкретные символы для исключения
# EXCLUDE_CATEGORIES: List[str] = []  # Категории: "whitespace", "control", "punctuation", "digits", "lowercase", "uppercase"
# INCLUDE_ONLY: List[str] = []  # Если не пустой, анализировать только эти символы (игнорирует EXCLUDE)

EXCLUDE_CHARS = ACTIVE_PRESET["EXCLUDE_CHARS"]
EXCLUDE_CATEGORIES = ACTIVE_PRESET["EXCLUDE_CATEGORIES"]
INCLUDE_ONLY = ACTIVE_PRESET["INCLUDE_ONLY"]


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

def compute_linearity(mask: np.ndarray) -> float:
    """PCA-based linearity: 1 = идеальная линия, 0 = облако точек."""
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return 0.0
    points = np.column_stack([xs, ys])
    pca = PCA(n_components=2)
    pca.fit(points)
    return float(pca.explained_variance_ratio_[0])

def compute_continuity(mask: np.ndarray) -> float:
    """Отношение площади наибольшей компоненты к общей площади."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return 0.0
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return float(max(sizes) / mask.sum())

def compute_orientation(mask: np.ndarray) -> float:
    """Угол главной оси в градусах (0=горизонталь, 90=вертикаль)."""
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return 0.0
    points = np.column_stack([xs, ys])
    pca = PCA(n_components=2)
    pca.fit(points)
    angle = np.degrees(np.arctan2(abs(pca.components_[0, 1]), 
                                   abs(pca.components_[0, 0])))
    return float(min(angle, 90 - angle))  # Нормализуем к [0, 45]


def compute_stroke_width(mask: np.ndarray) -> float:
    """Средняя толщина штриха в пикселях."""
    filled = int(mask.sum())
    if filled == 0:
        return 0.0
    try:
        skeleton = skeletonize(mask)
        skeleton_length = int(skeleton.sum())
        if skeleton_length == 0:
            return 0.0
        return filled / skeleton_length
    except Exception:
        return 0.0

def compute_branching(mask: np.ndarray) -> int:
    """Количество ветвлений в скелете символа."""
    try:
        skeleton = skeletonize(mask)
        # Ветвление = пиксель с 3+ соседями
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        neighbors = ndimage.convolve(skeleton.astype(int), kernel)
        branches = int(np.sum((skeleton) & (neighbors >= 3)))
        return branches
    except Exception:
        return 0

def compute_hatch_regularity(mask: np.ndarray) -> float:
    """Регулярность паттерна (автокорреляция проекции)."""
    proj = mask.sum(axis=0).astype(float)
    if proj.sum() == 0:
        return 0.0
    proj = proj - proj.mean()
    autocorr = np.correlate(proj, proj, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    if autocorr[0] == 0:
        return 0.0
    # Ищем первый пик после нуля (период)
    peaks = []
    for i in range(1, len(autocorr)):
        if autocorr[i] > autocorr[0] * 0.3:
            peaks.append(autocorr[i])
    return float(np.mean(peaks) / autocorr[0]) if peaks else 0.0

def compute_edge_contrast(mask: np.ndarray) -> float:
    """Контрастность краёв (градиент на границах)."""
    filled = int(mask.sum())
    if filled == 0:
        return 0.0
    # Размытие и сравнение с оригиналом
    blurred = ndimage.gaussian_filter(mask.astype(float), sigma=1.0)
    diff = np.abs(mask.astype(float) - blurred)
    return float(diff[mask].mean())

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
    return CharMetrics(
        brightness=float(filled),
        density=filled / total,
        holes=holes,
        components=n_components,
        euler_number=euler_number,
        # ... остальные поля ...
        char=ch,
        ord=code,
        category=category
    )