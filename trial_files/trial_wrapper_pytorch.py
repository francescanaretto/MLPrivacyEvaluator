"""
File to test the black box wrapper.
"""

import torch
import pandas as pd

from MLWrappers import PyTorchBlackBox

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




DS_NAME = 'adult'
DATA_FOLDER = f'./data/{DS_NAME}'
BB_PATH = f'./models/nn_torch_{DS_NAME}.pkl'

bb = PyTorchBlackBox(BB_PATH, Net(101, 2))

print(bb.model())

train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)
train_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_label.csv', skipinitialspace=True).to_numpy().ravel()
test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True)
test_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_label.csv', skipinitialspace=True).to_numpy().ravel()

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
