'''
In this file are contained the wrappers which implement the AbstractBbox class and methods.
If you don't find the wrapper for your model here, you can add yours by extending the
abstract class AbstractBbox.
'''

import pickle

from ._bbox import AbstractBBox


class SklearnBlackBox(AbstractBBox):
    """Wrapper for scikit-learn models."""

    def __init__(self, filename: str):
        with open(filename, 'rb') as file:
            self.bbox = pickle.load(file)

    def model(self):
        return self.bbox

    def predict(self, X):
        return self.bbox.predict(X)

    def predict_proba(self, X):
        return self.bbox.predict_proba(X)


class KerasBlackBox(AbstractBBox):
    """Wrapper for Keras neural network models."""

    def __init__(self):
        pass

    def model(self):
        pass


class PyTorchBlackBox(AbstractBBox):
    """Wrapper for PyTorch neural network models."""

    def __init__(self, filename: str):
        with open(filename, 'rb') as file:
            self.bbox = pickle.load(file)
        self.bbox.eval()

    def model(self):
        return self.bbox

    def predict(self, X):
        # TODO Implement pytorch predict
        pass

    def predict_proba(self, X):
        # TODO Implement pytorch predict probabilities
        pass
