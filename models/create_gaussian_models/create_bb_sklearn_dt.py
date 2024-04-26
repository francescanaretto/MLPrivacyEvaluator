"""
File to create a scikit-learn black box model.
"""

import pickle

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler


DS_NAME = 'gaussian'
DATA_FOLDER = f'./data/{DS_NAME}'
MODEL_TYPE = 'dt'
TO_UNDERSAMPLE = False

train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)
train_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_label.csv', skipinitialspace=True).to_numpy().ravel()
test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True)
test_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_label.csv', skipinitialspace=True).to_numpy().ravel()

print(train_set.shape)
print(train_label.shape)

model = DecisionTreeClassifier(min_samples_leaf=5, max_depth=8)
if TO_UNDERSAMPLE:
    undersampler = RandomUnderSampler(sampling_strategy='majority')
    train_set, train_label = undersampler.fit_resample(train_set, train_label)
model.fit(train_set.values, train_label)

train_pred = model.predict(train_set)
train_proba = model.predict_proba(train_set)

test_pred = model.predict(test_set)
test_proba = model.predict_proba(test_set)

print(classification_report(train_label, train_pred, digits=3))
print(classification_report(test_label, test_pred, digits=3))

with open(f'./models/{DS_NAME}_{MODEL_TYPE}.pkl', 'wb') as model_save_path:
    pickle.dump(model, model_save_path)
