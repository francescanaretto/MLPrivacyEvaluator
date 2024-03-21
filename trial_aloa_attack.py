"""
File to test the ALOA attack.
"""

import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from MLWrappers import SklearnBlackBox
from PrivacyAttacks import AloaPrivacyAttack

warnings.simplefilter("ignore", UserWarning)

DS_NAME = 'adult'
DATA_FOLDER = f'./data/{DS_NAME}'

N = 1

target = SklearnBlackBox(f'./models/rf_{DS_NAME}.sav')
attack = AloaPrivacyAttack(target, n_shadow_models=N)
# Passa nome per la cartella (dentro cartella shadow + attack)

train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)[:2000]
train_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_label.csv', skipinitialspace=True).to_numpy().ravel()
test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True) [:1000]
test_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_label.csv', skipinitialspace=True).to_numpy().ravel()
shadow_data = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_shadow_set.csv', skipinitialspace=True)

attack.fit(shadow_data)

in_set = np.full(train_set.shape[0], 'IN')
out_set = np.full(test_set.shape[0], 'OUT')

data = pd.concat([train_set, test_set])
membership = np.concatenate((in_set, out_set))

pred = attack.predict(data)
print(classification_report(membership, pred, digits=3))

print("Threshold", attack.attack_model.threshold)
