

from abc import ABC, abstractmethod, ABCMeta

import pandas as pd
import numpy as np


class AbstractBBox(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.array:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.array:
        pass
