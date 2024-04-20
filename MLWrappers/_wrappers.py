'''
This module contains the wrappers implementing the AbstractBBox class and methods.
If you do not find the wrapper for your model here, you can add your own wrapper by extending the
abstract class AbstractBbox.
'''

import pickle

import torch
from tensorflow import keras
import pandas as pd
import numpy as np

from ._bbox import AbstractBBox


class SklearnBlackBox(AbstractBBox):
    """Wrapper for scikit-learn models."""

    def __init__(self, filename: str):
        with open(filename, 'rb') as file:
            self.bbox = pickle.load(file)

    def model(self):
        return self.bbox

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.bbox.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.bbox.predict_proba(X)


class KerasBlackBox(AbstractBBox):
    """Wrapper for Keras neural network models."""

    def __init__(self, filename: str):
        self.bbox = keras.models.load_model(filename)

    def model(self):
        return self.bbox

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.bbox.predict(X, verbose=False)
        pred = np.argmax(pred, axis=1)
        return pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.bbox.predict(X, verbose=False)
        return proba


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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = torch.Tensor(X.values)
        pred = self.bbox(X).max(1)[1].numpy()
        return pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X = torch.Tensor(X)
        else:
            X = torch.Tensor(X.values)
        proba = self.bbox(X).detach().numpy()
        return proba
