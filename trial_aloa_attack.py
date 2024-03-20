"""
File to test the MIA attack.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.metrics import classification_report
from MLWrapper.wrappers import SklearnBlackBox
from PrivacyAttacks.aloa_privacy_attack import AloaPrivacyAttack

import warnings
warnings.simplefilter("ignore", UserWarning)

ds_name = 'gaussian'

n = 1

target = SklearnBlackBox(f'./models/rf_{ds_name}.sav')
attack = AloaPrivacyAttack(target, n_shadow_models=n)

train_data = pd.read_csv(f'./data/{ds_name}/{ds_name}_original_train_set.csv', skipinitialspace = True)[:2000]
train_label = pd.read_csv(f'./data/{ds_name}/{ds_name}_original_train_label.csv', skipinitialspace = True).to_numpy().ravel()
test_data = pd.read_csv(f'./data/{ds_name}/{ds_name}_original_test_set.csv', skipinitialspace = True)[:2000]
test_label = pd.read_csv(f'./data/{ds_name}/{ds_name}_original_test_label.csv', skipinitialspace = True).to_numpy().ravel()
shadow_data = pd.read_csv(f'./data/{ds_name}/{ds_name}_shadow_set.csv', skipinitialspace = True)

attack.fit(shadow_data)

in_set = np.full(train_data.shape[0], 'IN')
out_set = np.full(test_data.shape[0], 'OUT')

data = pd.concat([train_data, test_data])
membership = np.concatenate((in_set, out_set))

pred = attack.predict(data)
print(classification_report(membership, pred, digits=3))

print("Threshold", attack.attack_model.threshold)
