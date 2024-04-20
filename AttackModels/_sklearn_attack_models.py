"""
This module contains the implementation of attack models coming from the scikit-learn library.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ._attack_model import AttackModel


class AttackDecisionTree(AttackModel):
    """Attack model based on DecisionTreeClassifier."""

    def __init__(self, params):
        self.model = DecisionTreeClassifier(**params)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)


class AttackRandomForest(AttackModel):
    """Attack model based on RandomForestClassifier."""

    def __init__(self, params):
        self.model = RandomForestClassifier(**params)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)
