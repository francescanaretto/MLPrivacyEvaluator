from MLWrapper.bbox import AbstractBBox
from PrivacyAttacks.PrivacyAttack import *
import pandas as pd
from typing import *

'''
MLPrivacyEvaluator
This is the main class of the library, which can be used by users. With this class we can instantiate the main object necessary for testing the privacy of the Machine Learning models (and their data).
The init class contains the mandatory requirements needed to run the privacy attacks.'''
class MLPrivacyEvaluator():

    def __init__(self, black_box: AbstractBBox, train_set: pd.DataFrame, test_set: pd.DataFrame, train_labels: pd.DataFrame, test_labels: pd.DataFrame, privacy_attacks: list(PrivacyAttack)):
        self.bb = black_box
        self.train_set_bb = train_set
        self.train_labels_bb = train_labels
        self.test_set_bb = test_set
        self.test_labels_bb = test_labels
        self.privacy_attacks = privacy_attacks
