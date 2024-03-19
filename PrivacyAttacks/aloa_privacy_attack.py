
import pandas as pd
from PrivacyAttacks.privacy_attack import PrivacyAttack
from ShadowModels.random_forest_shadow_model import ShadowRandomForest


class AloaPrivacyAttack(PrivacyAttack):
    def __init__(self, black_box, shadow_model_type = 'rf'):
        super.__init__(black_box)
        self.shadow_model_type = shadow_model_type


    def _get_binary_features(self, X: pd.DataFrame):
        indices = []
        for i, column in enumerate(X):
            unique_values = set(X[column].unique())
            if unique_values == set([0, 1]):
                indices.append(i)
        return indices
    
    def _continuous_noise(self):
        pass

    def _binary_flip(self):
        pass

    def _get_shadow_model(self):
        if self.shadow_model_type == 'rf':
            shadow_model = ShadowRandomForest()
        return shadow_model

    def fit(self, shadow_dataset: pd.DataFrame):
        pass

    def predict(self):
        pass