
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ShadowModels.shadow_model import ShadowModel

class ShadowRandomForest(ShadowModel):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def fit(self, X: pd.DataFrame, y: np.array):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)

    def test(self):
        pass
