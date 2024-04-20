"""
Implementation of the general idea of a privacy attack, with abstract methods.
"""

from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from MLWrappers._bbox import AbstractBBox
from ShadowModels import ShadowRandomForest


class PrivacyAttack(ABC):

    def __init__(self, black_box: AbstractBBox,
                 shadow_model_type: str = 'rf'):
        self.bb = black_box
        self.shadow_model_type = shadow_model_type

    @abstractmethod
    def fit(self, shadow_dataset: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def _get_shadow_model(self):
        if self.shadow_model_type == 'rf':
            shadow_model = ShadowRandomForest()
        return shadow_model
