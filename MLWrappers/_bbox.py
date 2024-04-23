"""
The module contains the abstract class for black box model wrappers.
"""

from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class AbstractBBox(ABC):
    """Abstract class for black box model wrappers."""

    def __init__(self):
        pass

    @abstractmethod
    def model(self):
        """Returns the model object."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns the predicted labels for the input data."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns the predicted probability vectors for the input data."""
