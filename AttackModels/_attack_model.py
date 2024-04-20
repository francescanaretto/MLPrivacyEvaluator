"""
This module implements the abstract class for all attack models.
"""

from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class AttackModel(ABC):
    """Abstract class for attack models."""

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray):
        """Build the attack model from the training set (X, y)."""

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict the class of input samples X."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict the class probabilities of input samples X."""
