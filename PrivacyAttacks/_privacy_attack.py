"""
Implementation of the general idea of a privacy attack, with abstract methods.
"""

from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from MLWrappers._bbox import AbstractBBox
from ShadowModels import ShadowDecisionTree, ShadowRandomForest
from AttackModels import AttackDecisionTree, AttackRandomForest


class PrivacyAttack(ABC):
    """
    Abstract class for privacy attacks.

    Parameters
    ----------
    black_box : AbstractBBox
        The target machine learning model to be attacked.
    """

    def __init__(self, black_box: AbstractBBox,
                 shadow_model_type: str = 'rf',
                 shadow_model_params: dict = None,
                 attack_model_type: str = 'rf',
                 attack_model_params: dict = None):
        self.bb = black_box
        self.shadow_model_type = shadow_model_type
        if shadow_model_params is None:
            self.shadow_model_params = {}
        else:
            self.shadow_model_params = shadow_model_params
        self.attack_model_type = attack_model_type
        if attack_model_params is None:
            self.attack_model_params = {}
        else:
            self.attack_model_params = attack_model_params

    @abstractmethod
    def fit(self, shadow_dataset: pd.DataFrame):
        """Train the attack using the samples in shadow_dataset."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the attack predictions for the samples in X."""

    def _get_shadow_model(self):
        if self.shadow_model_type == 'rf':
            shadow_model = ShadowRandomForest(self.shadow_model_params)
        elif self.shadow_model_type == 'dt':
            shadow_model = ShadowDecisionTree(self.shadow_model_params)
        return shadow_model

    def _get_attack_model(self):
        if self.attack_model_type == 'rf':
            attack_model = AttackRandomForest(self.attack_model_params)
        elif self.attack_model_type == 'dt':
            attack_model = AttackDecisionTree(self.attack_model_params)
        return attack_model
