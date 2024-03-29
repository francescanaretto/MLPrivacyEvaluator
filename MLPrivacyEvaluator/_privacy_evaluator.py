

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from MLWrappers import AbstractBBox
from PrivacyAttacks import PrivacyAttack


class PrivacyEvaluator():
    """
    This is the main class of the library, which can be used by users. With this class we can instantiate the main
    object necessary for testing the privacy of the Machine Learning models (and their data).
    The init class contains the mandatory requirements needed to run the privacy attacks.
    """

    def __init__(self, black_box: AbstractBBox, privacy_attacks: list[PrivacyAttack]):
        self.bb = black_box
        self.privacy_attacks = privacy_attacks
        self.save_folder = None

    def fit(self, shadow_set: pd.DataFrame, save_files='all', save_folder='./default_save_folder'):
        self.save_folder = save_folder
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        for attack in self.privacy_attacks:
            attack.fit(shadow_set, save_files=save_files, save_folder=save_folder)

    def report(self, train_set: pd.DataFrame, test_set: pd.DataFrame, metrics='all'):
        results = {}

        in_set = np.full(train_set.shape[0], 'IN')
        out_set = np.full(test_set.shape[0], 'OUT')
        data = pd.concat([train_set, test_set])
        membership = np.concatenate((in_set, out_set))

        for attack in self.privacy_attacks:
            attack_res = {}
            predictions = attack.predict(data)
            attack_res['classification_report'] = classification_report(membership, predictions, digits=3, output_dict=True)
            results[attack.name] = attack_res

        return results
