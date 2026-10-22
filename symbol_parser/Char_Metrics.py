from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class CharMetrics:
    """Характеристики ASCII-符号а, оптимизированные для ASCII-арта."""
    
    # Основные
    brightness: float = 0.0
    density: float = 0.0
    
    # Топология
    holes: int = 0
    components: int = 0
    euler_number: int = 0
    
    # Структура и контуры (ASCII-специфичные)
    linearity: float = 0.0
    continuity: float = 0.0
    
    # Текстура и заливка
    hatch_regularity: float = 0.0
    block_var: float = 0.0
    edge_density: float = 0.0
    
    # Симметрия
    h_sym: float = 0.0
    v_sym: float = 0.0
    
    # Метаданные
    char: str = ""
    ord: int = 0
    category: str = ""
    
    # Динамические поля для визуализации
    pca_x: float = 0.0
    pca_y: float = 0.0

    @property
    def contour_score(self) -> float:
        """Пригодность для контуров."""
        return self.linearity * self.continuity * self.edge_density

    @property
    def fill_score(self) -> float:
        """Пригодность для равномерной заливки/дизеринга."""
        return (1 - self.linearity) * self.hatch_regularity * (1 - self.block_var)

    def describe(self) -> str:
        """Генерирует текстовое описание символа."""
        tags = []
        if self.holes > 0:
            tags.append(f"дырчатый ({self.holes})")
        else:
            tags.append("сплошной")
            
        if self.linearity > 0.7:
            tags.append("линейный")
        if self.continuity < 0.8 and self.components > 1:
            tags.append("разрозненный")
        if self.h_sym > 0.8 and self.v_sym > 0.8:
            tags.append("симметричный")
        if self.contour_score > 0.4:
            tags.append("контурный")
        if self.fill_score > 0.4:
            tags.append("текстурный")
            
        return ", ".join(tags)

    def to_dict(self) -> dict:
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