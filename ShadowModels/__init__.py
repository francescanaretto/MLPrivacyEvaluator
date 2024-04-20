"""
This module contains the shadow models used during
the training of privacy attacks.
"""

from ._sklearn_shadow_models import (
    ShadowDecisionTree,
    ShadowRandomForest
)


__all__ = [
    "ShadowDecisionTree",
    "ShadowRandomForest"
]
