'''
In this file are contained the wrappers which implement the AbstractBbox class and methods.
If you don't find the wrapper for your model here, you can add yours by extending the
abstract class AbstractBbox.
'''

import pickle

import torch
import pandas as pd

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

    def __init__(self, filename: str, nn_class=None):
        if nn_class is None:
            self.bbox = torch.jit.load(filename)
        else:
            self.bbox = nn_class
            self.bbox.load_state_dict(torch.load(filename))
        self.bbox.eval()

    def model(self):
        return self.bbox

    def predict(self, X: pd.DataFrame):
        X = torch.Tensor(X.values)
        pred = self.bbox(X).max(1)[1].numpy()
        return pred

    def predict_proba(self, X: pd.DataFrame):
        X = torch.Tensor(X.values)
        proba = self.bbox(X).detach().numpy()
        return proba
