from dataclasses import dataclass, asdict, field
from typing import ClassVar, Optional

@dataclass
class CharMetrics:
    """Характеристики ASCII-символа согласно требованиям v2.0"""
    
        # ===== СТАТИЧЕСКИЕ СПИСКИ МЕТРИК =====
    
    # Группы метрик по разделам требований v2.0
    GEOMETRY_METRICS: ClassVar[list] = [
        "brightness", "density",
        "aspect_ratio", "extent",
        "center_x", "center_y",
        "compactness", "solidity", "circularity",
    ]
    
    TOPOLOGY_METRICS: ClassVar[list] = [
        "components", "holes", "euler_number",
    ]
    
    ORIENTATION_METRICS: ClassVar[list] = [
        "h_var", "v_var",
        "pca_angle", "pca_ratio",
        "line_likeness",
    ]
    
    SYMMETRY_METRICS: ClassVar[list] = [
        "h_sym", "v_sym",
        "rot_sym_180", "symmetry_score",
    ]
    
    DETAIL_METRICS: ClassVar[list] = [
        "block_var", "edge_density",
        "local_variance", "texture_complexity",
    ]
    
    CONNECTIVITY_METRICS: ClassVar[list] = [
        "connectivity_4", "connectivity_8",
        "connectivity_ratio", "branching_points",
    ]
    
    # Все числовые метрики (для нормализации и кластеризации)
    ALL_NUMERIC_METRICS: ClassVar[list] = (
        GEOMETRY_METRICS +
        TOPOLOGY_METRICS +
        ORIENTATION_METRICS +
        SYMMETRY_METRICS +
        DETAIL_METRICS +
        CONNECTIVITY_METRICS
    )
    
    
    # Все метрики (включая категориальные)
    ALL_METRICS: ClassVar[list] = ALL_NUMERIC_METRICS + ["orientation"]
    
    # Метаданные (не метрики)
    METADATA_FIELDS: ClassVar[list] = ["char", "ord", "category", "pca_x", "pca_y"]
    
    # 2.1 Геометрические метрики
    brightness: float = 0.0          # Средняя яркость (инвертированная)
    density: float = 0.0             # Относительная плотность
    aspect_ratio: float = 0.0        # Соотношение сторон bbox
    extent: float = 0.0              # Заполненность bbox
    center_x: float = 0.0            # Нормализованная координата X центра масс
    center_y: float = 0.0            # Нормализованная координата Y центра масс
    compactness: float = 0.0         # Близость к кругу
    solidity: float = 0.0            # Выпуклость
    circularity: float = 0.0         # Округлость
    
    # 2.2 Топологические метрики
    components: int = 0              # Связные компоненты
    holes: int = 0                   # Дыры
    euler_number: int = 0            # Эйлерова характеристика
    
    # 2.3 Метрики направленности
    h_var: float = 0.0               # Дисперсия горизонтальной проекции
    v_var: float = 0.0               # Дисперсия вертикальной проекции
    pca_angle: float = 0.0           # Угол главной оси [-90°, 90°]
    pca_ratio: float = 0.0           # Отношение малого собственного значения к большому
    line_likeness: float = 0.0       # Линейность (НОВАЯ)
    orientation: str = "isotropic"   # Доминирующее направление
    
    # 2.4 Метрики симметрии
    h_sym: float = 0.0               # Горизонтальная симметрия
    v_sym: float = 0.0               # Вертикальная симметрия
    rot_sym_180: float = 0.0         # Симметрия при повороте на 180° (НОВАЯ)
    symmetry_score: float = 0.0      # Общая симметричность (НОВАЯ)
    
    # 2.5 Метрики детализации
    block_var: float = 0.0           # Блочная дисперсия
    edge_density: float = 0.0        # Плотность краёв
    local_variance: float = 0.0      # Локальная неоднородность (НОВАЯ)
    texture_complexity: float = 0.0  # Сложность текстуры (НОВАЯ)
    
    # 2.6 Метрики связности
    connectivity_4: int = 0          # 4-связные компоненты (НОВАЯ)
    connectivity_8: int = 0          # 8-связные компоненты (НОВАЯ)
    connectivity_ratio: float = 0.0  # Оценка диагональных связей (НОВАЯ)
    branching_points: int = 0        # Точки ветвления (НОВАЯ)
    
    # Устаревшие метрики (оставляем для обратной совместимости)
    linearity: float = 0.0
    """Устаревшая метрика"""
    continuity: float = 0.0
    """Устаревшая метрика"""
    hatch_regularity: float = 0.0
    """Устаревшая метрика"""
    
    # Метаданные
    char: str = ""
    ord: int = 0
    category: str = ""
    
    # Динамические поля для визуализации
    pca_x: float = 0.0
    pca_y: float = 0.0
        
    # ===== МЕТОДЫ КЛАССА =====
    
    @classmethod
    def get_metric_names(cls, category: str = "all") -> list:
        """
        Возвращает список названий метрик.
        
        Args:
            category: категория метрик. Варианты:
                - "all" — все метрики
                - "geometry" — геометрические
                - "topology" — топологические
                - "orientation" — направленность
                - "symmetry" — симметрия
                - "detail" — детализация
                - "connectivity" — связность
                - "numeric" — все числовые (для нормализации)
                - "metadata" — метаданные (char, ord, category)
        
        Returns:
            Список названий метрик
        """
        categories = {
            "all": cls.ALL_METRICS,
            "geometry": cls.GEOMETRY_METRICS,
            "topology": cls.TOPOLOGY_METRICS,
            "orientation": cls.ORIENTATION_METRICS,
            "symmetry": cls.SYMMETRY_METRICS,
            "detail": cls.DETAIL_METRICS,
            "connectivity": cls.CONNECTIVITY_METRICS,
            "numeric": cls.ALL_NUMERIC_METRICS,
            "metadata": cls.METADATA_FIELDS,
        }
        
        if category not in categories:
            raise ValueError(
                f"Неизвестная категория '{category}'. "
                f"Доступные: {list(categories.keys())}"
            )
        
        return categories[category]
    
    @classmethod
    def get_metrics_dict(cls, char_data: "CharMetrics") -> dict:
        """
        Возвращает словарь {имя_метрики: значение} для всех метрик.
        Удобно для экспорта и анализа.
        """
        return {
            name: getattr(char_data, name)
            for name in cls.ALL_METRICS
        }
    
    @classmethod
    def get_numeric_values(cls, char_data: "CharMetrics") -> list:
        """
        Возвращает список значений всех числовых метрик.
        Удобно для кластеризации и нормализации.
        """
        return [getattr(char_data, name) for name in cls.ALL_NUMERIC_METRICS]
    
    # Композитные метрики
    @property
    def contour_score(self) -> float:
        """Пригодность для контуров."""
        return self.line_likeness * self.symmetry_score * self.edge_density
    
    @property
    def fill_score(self) -> float:
        """Пригодность для заливки."""
        return (1 - self.line_likeness) * self.symmetry_score * (1 - self.block_var)
    
    def describe(self) -> str:
        """Генерирует текстовое описание символа."""
        tags = []
        
        if self.holes > 0:
            tags.append(f"дырчатый ({self.holes})")
        else:
            tags.append("сплошной")
        
        if self.line_likeness > 0.7:
            tags.append(f"линейный ({self.orientation})")
        elif self.line_likeness < 0.3:
            tags.append("округлый/точечный")
        
        if self.symmetry_score > 0.8:
            tags.append("симметричный")
        
        if self.texture_complexity > 0.5:
            tags.append("текстурный")
        
        return ", ".join(tags)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CharMetrics":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def classify_role(self) -> str:
        """
        Определяет функциональную роль символа (согласно разделу 3.3 требований.)
        
        Returns:
            Одна из ролей: 'contour', 'hatching', 'fill', 'dithering', 'mixed'
        """
        # Палитра контуров (3.3.1)
        if self.line_likeness > 0.7:
            return "contour"
        
        # Палитра заливки (3.3.3)
        if (self.line_likeness < 0.3 and 
            self.symmetry_score >= 0.7 and 
            self.block_var <= 0.15):
            return "fill"
        
        # Палитра дизеринга (3.3.4)
        if self.texture_complexity <= 0.3 and self.brightness < 0.3:
            return "dithering"
        
        # Палитра штриховки (3.3.2)
        if 0.5 <= self.line_likeness <= 0.8:
            return "hatching"
        
        return "mixed"


    def get_orientation_group(self) -> str:
        """Группирует символы по ориентации для палитры штриховки."""
        if self.orientation == "vertical":
            return "vertical"
        elif self.orientation == "horizontal":
            return "horizontal"
        elif self.orientation == "diagonal":
            return "diagonal"
        return "isotropic"
    
    def similarity_to(self, other: "CharMetrics", weights: Optional[dict] = None) -> float:
        """
        Вычисляет схожесть с другим символом.
        
        Args:
            other: другой символ
            weights: веса метрик (по умолчанию все равны)
        
        Returns:
            0.0 (полностью разные) - 1.0 (идентичные)
        """
        if weights is None:
            weights = {
                "brightness": 2.0,
                "holes": 1.5,
                "line_likeness": 1.0,
                "edge_density": 1.0,
                "block_var": 0.5,
            }
        
        total_weight = sum(weights.values())
        similarity = 0.0
        
        for metric, weight in weights.items():
            val1 = getattr(self, metric, 0.0)
            val2 = getattr(other, metric, 0.0)
            diff = abs(val1 - val2)
            similarity += (1 - diff) * weight
        
        return similarity / total_weight