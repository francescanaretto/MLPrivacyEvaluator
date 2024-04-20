"""
This module contains the implementation of shadow models coming from the scikit-learn library.

"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ._shadow_model import ShadowModel


class ShadowDecisionTree(ShadowModel):

    def __init__(self, **params):
        self.model = DecisionTreeClassifier(**params)

    def fit(self, X: pd.DataFrame, y: np.array):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)

    def test(self):
        pass


class ShadowRandomForest(ShadowModel):

    def __init__(self, **params):
        self.model = RandomForestClassifier(**params)

    def fit(self, X: pd.DataFrame, y: np.array):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)

    def test(self):
        pass
