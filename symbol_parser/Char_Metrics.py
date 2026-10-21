from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class CharMetrics:
    """Характеристики ASCII-символа."""
    
    # Основные метрики
    brightness: float = 0.0
    density: float = 0.0
    
    # Геометрия
    aspect: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    
    # Форма (новые метрики)
    solidity: float = 0.0
    extent: float = 0.0
    circularity: float = 0.0
    
    # Топология
    holes: int = 0
    components: int = 0
    euler_number: int = 0
    
    # Направленность
    h_var: float = 0.0
    v_var: float = 0.0
    
    # Текстура
    block_var: float = 0.0
    edge_density: float = 0.0
    edge_contrast: float = 0.0
    hatch_regularity: float = 0.0
    
    # Симметрия
    h_sym: float = 0.0
    v_sym: float = 0.0
    
    # Новые метрики для ASCII-арта
    linearity: float = 0.0
    continuity: float = 0.0
    orientation: float = 0.0
    stroke_width: float = 0.0
    branching: int = 0
    
    # Метаданные
    char: str = ""
    ord: int = 0
    category: str = ""
    
    # Композитные метрики (вычисляемые свойства)
    @property
    def contour_score(self) -> float:
        """Пригодность для контуров."""
        return self.linearity * self.continuity * self.edge_contrast
    
    @property
    def fill_score(self) -> float:
        """Пригодность для заливки."""
        return (1 - self.linearity) * self.hatch_regularity * (1 - self.block_var)
    
    @property
    def is_hollow(self) -> bool:
        """Имеет ли символ дырки."""
        return self.holes > 0
    
    @property
    def is_symmetric(self) -> bool:
        """Симметричен ли символ."""
        return self.h_sym > 0.85 and self.v_sym > 0.85
    
    @property
    def orientation_label(self) -> str:
        """Текстовое описание ориентации."""
        if self.h_var > self.v_var * 1.5:
            return "horizontal"
        elif self.v_var > self.h_var * 1.5:
            return "vertical"
        return "neutral"
    
    def describe(self) -> str:
        """Генерирует текстовое описание символа."""
        tags = []
        
        if self.is_hollow:
            tags.append(f"дырчатый ({self.holes} дырок)")
        else:
            tags.append("сплошной")
        
        if self.is_symmetric:
            tags.append("симметричный")
        
        if self.linearity > 0.8:
            tags.append(f"линейный ({self.orientation_label})")
        
        if self.contour_score > 0.5:
            tags.append("хорош для контуров")
        
        if self.fill_score > 0.5:
            tags.append("хорош для заливки")
        
        return ", ".join(tags)
    
    def to_dict(self) -> dict:
        """Конвертация в словарь для JSON."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CharMetrics":
        """Создание из словаря."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
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
                "brightness": 2.0,  # Яркость важна
                "holes": 1.5,       # Топология важна
                "linearity": 1.0,
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