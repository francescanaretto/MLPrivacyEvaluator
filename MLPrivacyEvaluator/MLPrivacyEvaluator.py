

from typing import *

import pandas as pd

from MLWrappers import AbstractBBox
from PrivacyAttacks import PrivacyAttack


class MLPrivacyEvaluator():
    """
    This is the main class of the library, which can be used by users. With this class we can instantiate the main
    object necessary for testing the privacy of the Machine Learning models (and their data).
    The init class contains the mandatory requirements needed to run the privacy attacks.
    """

    def __init__(self, black_box: AbstractBBox, privacy_attacks: list[PrivacyAttack]):
        self.bb = black_box
        self.privacy_attacks = privacy_attacks

    def fit(self, shadow_set: pd.DataFrame, save_files='all'):
        # TODO Implement the execution of the attacks
        pass

    def report(self, train_set: pd.DataFrame, test_set: pd.DataFrame, metrics='all'):
        # TODO Implement the reporting
        pass
