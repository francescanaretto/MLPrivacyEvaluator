from abc import ABC, abstractmethod


class AttackModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def predict_proba(self):
        pass
