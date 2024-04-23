"""
File to test the MIA attack.
"""

import warnings

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from MLWrappers import SklearnBlackBox, PyTorchBlackBox, KerasBlackBox
from PrivacyAttacks import MiaPrivacyAttack

warnings.simplefilter("ignore", UserWarning)

DS_NAME = 'gaussian'
DATA_FOLDER = f'./data/{DS_NAME}'

for n in range(3, 4):
    target = SklearnBlackBox(f'./models/rf_{DS_NAME}.pkl')
    # target = PyTorchBlackBox(f'./models/nn_torch_{DS_NAME}.pt')
    # target = KerasBlackBox(f'./models/nn_keras_{DS_NAME}.keras')

    attack = MiaPrivacyAttack(target, n_shadow_models=n, voting_model=False)

    train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)
    train_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_label.csv', skipinitialspace=True).to_numpy().ravel()
    test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True)
    test_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_label.csv', skipinitialspace=True).to_numpy().ravel()
    shadow_data = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_shadow_set.csv', skipinitialspace=True)

    attack.fit(shadow_data)

    in_set = np.full(train_set.shape[0], 'IN')
    out_set = np.full(test_set.shape[0], 'OUT')

    data = pd.concat([train_set, test_set])
    membership = np.concatenate((in_set, out_set))

    pred = attack.predict(data)
    print(f" ******  WITH {n} SHADOW MODELS  ****** ")
    print(classification_report(membership, pred, digits=3))
