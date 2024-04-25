"""
File to test the MIA attack.
"""

import warnings

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from MLWrappers import SklearnBlackBox
from PrivacyAttacks import MiaPrivacyAttack

warnings.simplefilter("ignore", UserWarning)

DS_NAME = 'adult'
DATA_FOLDER = f'./data/{DS_NAME}'

target = SklearnBlackBox(f'./models/{DS_NAME}_rf.pkl')
# target = PyTorchBlackBox(f'./models/{DS_NAME}_nn_torch.pt')
# target = KerasBlackBox(f'./models/{DS_NAME}_nn_keras.keras')

attack = MiaPrivacyAttack(target, n_shadow_models=3, voting_model=False,
                          shadow_model_type='rf',
                          shadow_model_params={'n_estimators': 3})

train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)[:1000]
train_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_label.csv', skipinitialspace=True).to_numpy().ravel()
test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True)[:1000]
test_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_label.csv', skipinitialspace=True).to_numpy().ravel()
shadow_data = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_shadow_set.csv', skipinitialspace=True)[:1000]

attack.fit(shadow_data)

in_set = np.full(train_set.shape[0], 'IN')
out_set = np.full(test_set.shape[0], 'OUT')

data = pd.concat([train_set, test_set])
membership = np.concatenate((in_set, out_set))

pred = attack.predict(data)
print(classification_report(membership, pred, digits=3))
