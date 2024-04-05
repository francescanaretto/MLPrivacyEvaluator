

import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report

from ._attack_model import AttackModel


class AttackThresholdModel(AttackModel):

    def __init__(self):
        self.threshold = None

    def _get_score(self, score_type, y, th_data):
        if score_type == 'accuracy':
            return accuracy_score(y, th_data)
        elif score_type == 'precision':
            return precision_score(y, th_data)
        elif score_type == 'recall':
            return recall_score(y, th_data)
        else:
            raise ValueError('Invalid score type entered. Please use "accuracy", "precision" or "recall".')

    def fit(self, x: np.array, y: np.array, thresholds: np.array = None, score_type: str = 'accuracy'):
        results = []
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)
        for t in thresholds:
            # th_data = np.array(list(map(lambda value: 0 if value <= t else 1, x)))
            th_data = np.array([1 if value > t else 0 for value in x])
            score = self._get_score(score_type, y, th_data)
            results.append(score)
            # print(classification_report(y, th_data))
        threshold_chosen = thresholds[np.argmax(results)]
        self.threshold = threshold_chosen
        return self.threshold

    def predict(self, x: np.array):
        predictions = np.array(list(map(lambda value: 0 if value <= self.threshold else 1, x)))
        return predictions

    def predict_proba(self):
        pass
