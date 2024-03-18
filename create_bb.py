"""
File to create a black box model.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DS_NAME = 'adult'
MODEL_TYPE = 'rf'

train_set = pd.read_csv(f'./data/{DS_NAME}/{DS_NAME}_original_train_set.csv', skipinitialspace = True)
train_label = pd.read_csv(f'./data/{DS_NAME}/{DS_NAME}_original_train_label.csv', skipinitialspace = True).to_numpy().ravel()
test_set = pd.read_csv(f'./data/{DS_NAME}/{DS_NAME}_original_test_set.csv', skipinitialspace = True)
test_label = pd.read_csv(f'./data/{DS_NAME}/{DS_NAME}_original_test_label.csv', skipinitialspace = True).to_numpy().ravel()


print(train_set.shape)
print(train_label.shape)


model = RandomForestClassifier(n_estimators=100)
model.fit(train_set.values, train_label)

train_pred = model.predict(train_set)
train_proba = model.predict_proba(train_set)

test_pred = model.predict(test_set)
test_proba = model.predict_proba(test_set)

print(classification_report(train_label, train_pred))
print(classification_report(test_label, test_pred))

pickle.dump(model, open(f'./models/{MODEL_TYPE}_{DS_NAME}.sav', 'wb'))
