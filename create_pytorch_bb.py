"""
File to create a PyTorch black box model.
"""

import pickle

import torch
import pandas as pd
from sklearn.metrics import classification_report


class Net(torch.nn.Module):
    """PyTorch neural network."""

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.fc1 = torch.nn.Linear(self.n_features, 64)
        self.fc99 = torch.nn.Linear(64, self.n_classes)
        self.dropout = torch.nn.Dropout(0.30)

    def forward(self, x):
        x = torch.nn.functional.tanh(self.fc1(x))
        x = self.dropout(x)
        out = torch.nn.functional.softmax(self.fc99(x), dim=1)
        return out


DS_NAME = 'gaussian'
DATA_FOLDER = f'./data/{DS_NAME}'
MODEL_TYPE = 'nn_torch'

train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)
train_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_label.csv', skipinitialspace=True).to_numpy().ravel()
test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True)
test_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_label.csv', skipinitialspace=True).to_numpy().ravel()

print(train_set.shape)
print(train_label.shape)

model = Net(n_features=10, n_classes=2)
model.fit(train_set.values, train_label)

train_pred = model.predict(train_set)
train_proba = model.predict_proba(train_set)

test_pred = model.predict(test_set)
test_proba = model.predict_proba(test_set)

print(classification_report(train_label, train_pred))
print(classification_report(test_label, test_pred))

with open(f'./models/{MODEL_TYPE}_{DS_NAME}.pkl', 'wb') as model_save_path:
    pickle.dump(model, model_save_path)
