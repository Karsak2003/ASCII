#region Импорты и типы

import numpy as np
from typing import Dict, List, Any, Tuple


from scipy import ndimage
from scipy.spatial import ConvexHull

from sklearn.decomposition import PCA

from skimage.morphology import skeletonize

from char_metrics import CharMetrics

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
    "EXCLUDE_CHARS": [],
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
    # Геометрия
    "brightness", "density", "extent", "solidity", "circularity",
    
    # Топология
    "holes", "components", "euler_number",
    
    # Направленность
    "line_likeness", "pca_ratio",
    
    # Симметрия
    "symmetry_score", "rot_sym_180",
    
    # Детализация
    "block_var", "edge_density", "texture_complexity",
    
    # Связность
    "connectivity_ratio", "branching_points",
]
#endregion Параметры и константы


def compute_solidity(mask: np.ndarray) -> float:
    """Вычисляет solidity (отношение площади к площади выпуклой оболочки)."""
    filled = int(mask.sum())
    if filled == 0:
        return 0.0
    
    ys, xs = np.where(mask)
    points = np.column_stack([xs, ys])
    
    if len(points) < 3:
        return 1.0
    
    try:
        hull = ConvexHull(points)
        hull_area = float(hull.volume)
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

def compute_metrics(mask: np.ndarray, ch: str, code: int, category: str) -> CharMetrics:
    """Вычисляет все метрики для бинарной маски согласно требованиям v2.0."""
    filled = int(mask.sum())
    total = mask.size
    
    if filled == 0:
        return CharMetrics(char=ch, ord=code, category=category)
    
    # === 2.1 Геометрические метрики ===
    # Bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_w = cmax - cmin + 1
    bbox_h = rmax - rmin + 1
    
    aspect_ratio = float(bbox_w / bbox_h) if bbox_h else 0.0
    extent = float(filled / (bbox_w * bbox_h)) if (bbox_w * bbox_h) > 0 else 0.0
    
    # Центр масс
    ys, xs = np.where(mask)
    center_x = float(xs.mean()) / CELL_W
    center_y = float(ys.mean()) / CELL_H
    
    # Края для compactness и edge_density
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    edges = ndimage.convolve(mask.astype(int), kernel) != 0
    perimeter = int(edges.sum())
    
    compactness = float((perimeter ** 2) / (4 * np.pi * filled)) if filled > 0 else 0.0
    circularity = float(4 * np.pi * filled / (perimeter ** 2)) if perimeter > 0 else 0.0
    solidity = compute_solidity(mask)
    
    # === 2.2 Топологические метрики ===
    labeled, n_components = ndimage.label(mask)
    bbox_mask = np.zeros_like(mask)
    bbox_mask[rmin:rmax+1, cmin:cmax+1] = True
    bg = (~mask) & bbox_mask
    holes = 0
    if bg.any():
        bg_labeled, n_bg = ndimage.label(bg)
        for lbl in range(1, n_bg + 1):
            comp = bg_labeled == lbl
            if not (comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()):
                holes += 1
    
    euler_number = n_components - holes
    
    # === 2.3 Метрики направленности ===
    h_proj = mask.sum(axis=1).astype(float)
    v_proj = mask.sum(axis=0).astype(float)
    h_var = float(h_proj.var()) / (filled ** 2)
    v_var = float(v_proj.var()) / (filled ** 2)
    
    pca_angle, pca_ratio, line_likeness = compute_pca_metrics(mask)
    orientation = compute_orientation(h_var, v_var, pca_angle, pca_ratio)
    
    # === 2.4 Метрики симметрии ===
    h_sym = float(np.minimum(mask, mask[:, ::-1]).sum()) / filled
    v_sym = float(np.minimum(mask, mask[::-1, :]).sum()) / filled
    rot_sym_180 = compute_rot_symmetry_180(mask)
    symmetry_score = (h_sym + v_sym) / 2
    
    # === 2.5 Метрики детализации ===
    mid_y, mid_x = CELL_H // 2, CELL_W // 2
    blocks = [
        float(mask[:mid_y, :mid_x].sum()),
        float(mask[:mid_y, mid_x:].sum()),
        float(mask[mid_y:, :mid_x].sum()),
        float(mask[mid_y:, mid_x:].sum()),
    ]
    block_var = float(np.var(blocks)) / (filled ** 2)
    
    edge_density = float(edges.sum()) / filled
    local_variance = compute_local_variance(mask)
    texture_complexity = compute_texture_complexity(edge_density, local_variance)
    
    # === 2.6 Метрики связности ===
    connectivity_4, connectivity_8, connectivity_ratio = compute_connectivity(mask)
    branching_points = compute_branching_points(mask)
    
    return CharMetrics(
        # 2.1 Геометрия
        brightness=float(filled),
        density=float(filled / total),
        aspect_ratio=aspect_ratio,
        extent=extent,
        center_x=center_x,
        center_y=center_y,
        compactness=compactness,
        solidity=solidity,
        circularity=circularity,
        
        # 2.2 Топология
        components=n_components,
        holes=holes,
        euler_number=euler_number,
        
        # 2.3 Направленность
        h_var=h_var,
        v_var=v_var,
        pca_angle=pca_angle,
        pca_ratio=pca_ratio,
        line_likeness=line_likeness,
        orientation=orientation,
        
        # 2.4 Симметрия
        h_sym=h_sym,
        v_sym=v_sym,
        rot_sym_180=rot_sym_180,
        symmetry_score=symmetry_score,
        
        # 2.5 Детализация
        block_var=block_var,
        edge_density=edge_density,
        local_variance=local_variance,
        texture_complexity=texture_complexity,
        
        # 2.6 Связность
        connectivity_4=connectivity_4,
        connectivity_8=connectivity_8,
        connectivity_ratio=connectivity_ratio,
        branching_points=branching_points,
        
        # Метаданные
        char=ch,
        ord=code,
        category=category
    )

def compute_pca_metrics(mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Вычисляет PCA-метрики: угол, соотношение, линейность.
    
    Returns:
        (pca_angle, pca_ratio, line_likeness)
    """
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return 0.0, 1.0, 0.0
    
    points = np.column_stack([xs, ys])
    pca = PCA(n_components=2)
    pca.fit(points)
    
    # Угол главной оси
    angle = np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0]))
    
    # Соотношение собственного значения
    eigenvalues = pca.explained_variance_
    ratio = eigenvalues[1] / eigenvalues[0] if eigenvalues[0] > 0 else 1.0
    
    # Line-likeness: (1 - pca_ratio) × solidity × (1 - circularity)
    solidity = compute_solidity(mask)
    circularity = compute_circularity(mask)
    line_likeness = (1 - ratio) * solidity * (1 - circularity)
    
    return float(angle), float(ratio), float(line_likeness)


def compute_orientation(h_var: float, v_var: float, pca_angle: float, pca_ratio: float) -> str:
    """Определяет доминирующее направление."""
    if v_var > h_var * 1.5:
        return "vertical"
    elif h_var > v_var * 1.5:
        return "horizontal"
    elif abs(h_var - v_var) < 0.3 * max(h_var, v_var) and abs(pca_angle) > 30:
        return "diagonal"
    elif pca_ratio > 0.7:
        return "isotropic"
    return "mixed"


def compute_rot_symmetry_180(mask: np.ndarray) -> float:
    """Симметрия при повороте на 180°."""
    filled = int(mask.sum())
    if filled == 0:
        return 0.0
    rotated = np.rot90(mask, 2)
    overlap = np.minimum(mask, rotated).sum()
    return float(overlap) / filled


def compute_local_variance(mask: np.ndarray) -> float:
    """Локальная неоднородность (дисперсия в окне 3×3)."""
    filled = int(mask.sum())
    if filled == 0:
        return 0.0
    
    # Скользящее окно 3×3
    kernel = np.ones((3, 3)) / 9
    local_mean = ndimage.uniform_filter(mask.astype(float), size=3)
    local_var = np.var(local_mean[mask])
    
    return float(local_var)


def compute_texture_complexity(edge_density: float, local_variance: float) -> float:
    """Сложность текстуры: комбинация edge_density и local_variance."""
    return float(edge_density * 0.6 + local_variance * 0.4)


def compute_connectivity(mask: np.ndarray) -> Tuple[int, int, float]:
    """
    Вычисляет метрики связности.
    
    Returns:
        (connectivity_4, connectivity_8, connectivity_ratio)
    """
    # 4-связность
    structure_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled_4, n_4 = ndimage.label(mask, structure=structure_4)
    
    # 8-связность
    structure_8 = np.ones((3, 3))
    labeled_8, n_8 = ndimage.label(mask, structure=structure_8)
    
    ratio = n_8 / n_4 if n_4 > 0 else 1.0
    
    return int(n_4), int(n_8), float(ratio)


def compute_branching_points(mask: np.ndarray) -> int:
    """Количество точек ветвления (пиксели с 3+ соседями)."""
    try:
        skeleton = skeletonize(mask)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbors = ndimage.convolve(skeleton.astype(int), kernel)
        return int(np.sum((skeleton) & (neighbors >= 3)))
    except Exception:
        return 0

import csv
from pathlib import Path


def export_to_csv(data: List[CharMetrics], output_path: str) -> None:
    """Экспортирует метрики в CSV для Excel/LibreOffice."""
    if not data:
        return
    
    # Все поля dataclass в нужном порядке
    fieldnames = CharMetrics.METADATA_FIELDS + CharMetrics.ALL_METRICS
    
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for c in data:
            row = {**c.to_dict(), "role": c.classify_role()}
            # Фильтруем словарь, оставляя только нужные поля
            filtered_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered_row)
    
    print(f"[i] CSV сохранён: {output_path}")


def export_to_html(data: List[CharMetrics], output_path: str) -> None:
    """
    Создаёт интерактивную HTML-таблицу с фильтрами и сортировкой.
    Открывается в браузере, поддерживает поиск, фильтры по колонкам, сортировку.
    """
    # Добавляем классификацию
    for c in data:
        c.role = c.classify_role()
        c.orientation_group = c.get_orientation_group()
    
    fieldnames = [
        "char", "ord", "category", "role", "orientation_group",
        "brightness", "density",
        "line_likeness", "orientation",
        "symmetry_score", "block_var", "texture_complexity",
        "holes", "euler_number",
        "contour_score", "fill_score",
        "h_var", "v_var", "pca_angle",
        "edge_density", "extent", "solidity",
    ]
    
    html = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>ASCII Symbol Metrics</title>
<style>
  body { font-family: Consolas, monospace; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
  h1 { color: #569cd6; }
  .controls { margin-bottom: 20px; display: flex; gap: 10px; flex-wrap: wrap; }
  input, select { background: #2d2d2d; color: #d4d4d4; border: 1px solid #3e3e3e; padding: 6px 10px; border-radius: 4px; font-family: inherit; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th { background: #2d2d2d; color: #569cd6; position: sticky; top: 0; cursor: pointer; user-select: none; padding: 8px; border: 1px solid #3e3e3e; }
  th:hover { background: #3e3e3e; }
  td { padding: 4px 8px; border: 1px solid #3e3e3e; text-align: center; }
  tr:hover { background: #2d2d2d; }
  .char-cell { font-size: 18px; font-weight: bold; }
  .role-contour { color: #4ec9b0; }
  .role-hatching { color: #ce9178; }
  .role-fill { color: #dcdcaa; }
  .role-dithering { color: #c586c0; }
  .role-mixed { color: #808080; }
  .stats { margin-top: 20px; padding: 10px; background: #2d2d2d; border-radius: 4px; }
</style>
</head>
<body>
<h1>ASCII Symbol Metrics — Interactive Table</h1>
<div class="controls">
  <input type="text" id="search" placeholder="Поиск по символу...">
  <select id="roleFilter">
    <option value="">Все роли</option>
    <option value="contour">Контурные</option>
    <option value="hatching">Штриховка</option>
    <option value="fill">Заливка</option>
    <option value="dithering">Дизеринг</option>
    <option value="mixed">Смешанные</option>
  </select>
  <select id="categoryFilter">
    <option value="">Все категории</option>
    <option value="punctuation">Punctuation</option>
    <option value="digits">Digits</option>
    <option value="uppercase">Uppercase</option>
    <option value="lowercase">Lowercase</option>
  </select>
  <select id="orientationFilter">
    <option value="">Все ориентации</option>
    <option value="vertical">Вертикальные</option>
    <option value="horizontal">Горизонтальные</option>
    <option value="diagonal">Диагональные</option>
    <option value="isotropic">Изотропные</option>
  </select>
</div>
<div class="stats" id="stats"></div>
<table id="metricsTable">
  <thead><tr>"""
    
    for fn in fieldnames:
        html += f"<th onclick=\"sortTable('{fn}')\">{fn}</th>"
    
    html += "</tr></thead><tbody id=\"tableBody\">"
    
    for c in data:
        row = c.to_dict()
        row["role"] = c.role
        row["orientation_group"] = c.orientation_group
        html += "<tr>"
        for fn in fieldnames:
            val = row.get(fn, "")
            cls = ""
            if fn == "char":
                cls = ' class="char-cell"'
            elif fn == "role":
                cls = f' class="role-{val}"'
            if isinstance(val, float):
                val = f"{val:.4f}"
            html += f"<td{cls}>{val}</td>"
        html += "</tr>"
    
    html += """</tbody></table>
<script>
let sortCol = '';
let sortAsc = true;

function sortTable(col) {
  if (sortCol === col) sortAsc = !sortAsc;
  else { sortCol = col; sortAsc = true; }
  applyFilters();
}

function applyFilters() {
  const search = document.getElementById('search').value.toLowerCase();
  const roleF = document.getElementById('roleFilter').value;
  const catF = document.getElementById('categoryFilter').value;
  const oriF = document.getElementById('orientationFilter').value;
  const rows = Array.from(document.querySelectorAll('#tableBody tr'));
  let visible = 0;
  
  rows.forEach(row => {
    const cells = row.querySelectorAll('td');
    const char = cells[0].textContent.toLowerCase();
    const role = cells[3].textContent;
    const cat = cells[2].textContent;
    const ori = cells[4].textContent;
    
    const matchSearch = char.includes(search);
    const matchRole = !roleF || role === roleF;
    const matchCat = !catF || cat === catF;
    const matchOri = !oriF || ori === oriF;
    
    if (matchSearch && matchRole && matchCat && matchOri) {
      row.style.display = '';
      visible++;
    } else {
      row.style.display = 'none';
    }
  });
  
  document.getElementById('stats').textContent = `Показано: ${visible} из ${rows.length} символов`;
  
  if (sortCol) {
    const idx = Array.from(document.querySelectorAll('th')).findIndex(th => th.textContent === sortCol);
    if (idx >= 0) {
      rows.sort((a, b) => {
        const va = a.children[idx].textContent;
        const vb = b.children[idx].textContent;
        const na = parseFloat(va), nb = parseFloat(vb);
        const cmp = isNaN(na) ? va.localeCompare(vb) : na - nb;
        return sortAsc ? cmp : -cmp;
      });
      const tbody = document.getElementById('tableBody');
      rows.forEach(r => tbody.appendChild(r));
    }
  }
}

document.getElementById('search').addEventListener('input', applyFilters);
document.getElementById('roleFilter').addEventListener('change', applyFilters);
document.getElementById('categoryFilter').addEventListener('change', applyFilters);
document.getElementById('orientationFilter').addEventListener('change', applyFilters);
applyFilters();
</script>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"[i] HTML-таблица сохранена: {output_path}")