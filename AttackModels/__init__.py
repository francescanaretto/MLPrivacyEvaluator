"""
This module contains the attack models used for
prediction.
"""

from ._random_forest_attack_model import AttackRandomForest
from ._threshold_attack_model import AttackThresholdModel

__all__ = [
    "AttackRandomForest",
    "AttackThresholdModel"
]
