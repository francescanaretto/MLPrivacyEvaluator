

from abc import ABC, abstractmethod

from MLWrappers._bbox import AbstractBBox


class PrivacyAttack(ABC):

    def __init__(self, black_box: AbstractBBox):
        self.bb = black_box

    @abstractmethod
    def fit(self, shadow_dataset):
        pass

    @abstractmethod
    def predict(self, X):
        pass
