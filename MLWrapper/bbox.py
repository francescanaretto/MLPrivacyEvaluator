from abc import ABC, abstractmethod, ABCMeta
import pandas as pd


class AbstractBBox(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame):
        pass
