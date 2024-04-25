"""
File to test dataset splitting in attack dataset generation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold,  KFold


DS_NAME = 'adult'
DATA_FOLDER = f'./data/{DS_NAME}'

train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)
train_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_label.csv', skipinitialspace=True).to_numpy().ravel()
test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True)
test_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_label.csv', skipinitialspace=True).to_numpy().ravel()
shadow_data = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_shadow_set.csv', skipinitialspace=True)

N = 4

res = shadow_data.sample(frac=max(1/N, 0.2), replace=False)

print(res.shape)
print(np.array(res.index))
