"""
Implementation of the general idea of a privacy attack, with abstract methods.
"""

from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from MLWrappers._bbox import AbstractBBox
from ShadowModels import ShadowDecisionTree, ShadowRandomForest


class PrivacyAttack(ABC):

    def __init__(self, black_box: AbstractBBox,
                 shadow_model_type: str = 'rf',
                 shadow_model_params: dict = {}):
        self.bb = black_box
        self.shadow_model_type = shadow_model_type
        self.shadow_model_params = shadow_model_params

    @abstractmethod
    def fit(self, shadow_dataset: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def _get_shadow_model(self):
        if self.shadow_model_type == 'rf':
            shadow_model = ShadowRandomForest(self.shadow_model_params)
        elif self.shadow_model_type == 'dt':
            shadow_model = ShadowDecisionTree(self.shadow_model_params)
        return shadow_model
