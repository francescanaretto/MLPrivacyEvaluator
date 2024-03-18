"""
File to test the black box wrapper.
"""

import pandas as pd
from MLWrapper.wrappers import SklearnBlackBox

DS_NAME = 'adult'

filename = f'./models/rf_{DS_NAME}.sav'

bb = SklearnBlackBox(filename)

print(bb.model())

train_set = pd.read_csv(f'./data/{DS_NAME}/{DS_NAME}_original_train_set.csv', skipinitialspace = True)
train_label = pd.read_csv(f'./data/{DS_NAME}/{DS_NAME}_original_train_label.csv', skipinitialspace = True).to_numpy().ravel()
test_set = pd.read_csv(f'./data/{DS_NAME}/{DS_NAME}_original_test_set.csv', skipinitialspace = True)
test_label = pd.read_csv(f'./data/{DS_NAME}/{DS_NAME}_original_test_label.csv', skipinitialspace = True).to_numpy().ravel()

print("Label predictions.")
pred = bb.predict(train_set)
print(pred)
print(type(pred))
print(pred.shape)

print("Probability vectors.")
proba = bb.predict_proba(train_set)
print(proba)
print(type(proba))
print(proba.shape)
