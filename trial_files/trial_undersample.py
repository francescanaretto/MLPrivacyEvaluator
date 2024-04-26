"""
Script to test the undersampler.
"""

import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler


sampler = RandomUnderSampler(sampling_strategy='majority')

DS_NAME = 'adult'
DATA_FOLDER = f'./data/{DS_NAME}'

train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)
train_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_label.csv', skipinitialspace=True).to_numpy().ravel()
test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True)
test_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_label.csv', skipinitialspace=True).to_numpy().ravel()
shadow_data = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_shadow_set.csv', skipinitialspace=True)

train_set['class'] = train_label
print(train_set)

print(type(train_label))
print(train_label.shape)

train_set, train_label = sampler.fit_resample(train_set, train_label)

print(train_set)
print(train_label)

print(type(train_label))
print(train_label.shape)
