

from abc import ABC, abstractmethod, ABCMeta

import pandas as pd
import numpy as np


class AbstractBBox(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def model(self):
        """
        Returns the model object.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.array:
        """
        Returns the predicted labels for the input data.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """
        Returns the predicted probability vectors for the input data.
        """
        pass
