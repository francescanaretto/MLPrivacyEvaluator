from abc import ABC, abstractmethod, ABCMeta


class AbstractBBox(ABC):
    def __init__(self, classifier):
        self.bbox = classifier

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass
