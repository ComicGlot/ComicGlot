import numpy as np
from dataclasses import dataclass

@dataclass
class TextBlock:
    box: np.ndarray  # [x1, y1, x2, y2] format
    class_name: str
    confidence: float
    id: int = 0