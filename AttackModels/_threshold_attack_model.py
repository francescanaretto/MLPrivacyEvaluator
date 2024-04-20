"""
This module provides an implementation of the threshold model used in LabelOnly and ALOA attacks.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_curve

from ._attack_model import AttackModel


class AttackThresholdModel(AttackModel):
    """Threshold model used for the LabelOnly and ALOA attacks."""

    def __init__(self):
        self.threshold = None

    def fit(self, x: np.ndarray, y: np.ndarray, thresholds: np.ndarray | None = None, score_type: str = 'accuracy'):
        results = []
        if thresholds is None:
            _, _, thresholds = roc_curve(y, x, pos_label='IN')
        for t in thresholds:
            th_data = np.array(['IN' if value > t else 'OUT' for value in x])
            score = self._get_score(score_type, y, th_data)
            results.append(score)
        threshold_chosen = thresholds[np.argmax(results)]
        self.threshold = threshold_chosen
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.array(list(map(lambda value: 'OUT' if value <= self.threshold else 'IN', x)))
        return predictions

    def predict_proba(self, x: np.ndarray):
        pass

    def _get_score(self, score_type, y, th_data):
        if score_type == 'accuracy':
            return accuracy_score(y, th_data)
        elif score_type == 'precision':
            return precision_score(y, th_data)
        elif score_type == 'recall':
            return recall_score(y, th_data)
        else:
            raise ValueError('Invalid score type entered. Please use "accuracy", "precision" or "recall".')
