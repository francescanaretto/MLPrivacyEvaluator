'''
In this file are contained the wrappers which implement the AbstractBbox class and methods.
If you don't find the wrapper for your model here, you can add yours by extending the
abstract class AbstractBbox.
'''

import pickle

import pandas as pd

from MLWrapper.bbox import AbstractBBox


class SklearnBlackBox(AbstractBBox):
    """
    Wrapper for scikit-learn models, e.g., random forest, decision tree
    """

    def __init__(self, filename: str):
        with open(filename, 'rb') as file:
            self.bbox = pickle.load(file)

    def model(self):
        return self.bbox

    def predict(self, X: pd.DataFrame):
        return self.bbox.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.bbox.predict_proba(X)


class KerasBlackBox(AbstractBBox):

    def __init__(self):
        pass

    def model(self):
        pass

# TODO Add the wrappers for the pytorch models and the keras models.
