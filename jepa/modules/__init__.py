from .base import BaseModule
from .models.jepa import JEPA
from .models.semi_contrastive import SemiContrastiveLearning
from .models.true_contrastive import TrueContrastiveLearning

__all__ = [
    "JEPA",
    "SemiContrastiveLearning",
    "TrueContrastiveLearning",
]