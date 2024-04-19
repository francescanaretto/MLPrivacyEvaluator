

from abc import ABC, abstractmethod

from MLWrappers._bbox import AbstractBBox
from ShadowModels import ShadowRandomForest


class PrivacyAttack(ABC):

    def __init__(self, black_box: AbstractBBox, shadow_model_type='rf'):
        self.bb = black_box
        self.shadow_model_type = shadow_model_type

    @abstractmethod
    def fit(self, shadow_dataset):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def _get_shadow_model(self):
        if self.shadow_model_type == 'rf':
            shadow_model = ShadowRandomForest()
        return shadow_model
