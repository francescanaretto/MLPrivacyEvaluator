"""
File to create a PyTorch black box model.
"""

import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
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


class MyDataset(Dataset):

    def __init__(self, x, y):
        self.x_data =x
        self.y_data = y
        self.len = x.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


DS_NAME = 'gaussian'
DATA_FOLDER = f'./data/{DS_NAME}'
MODEL_TYPE = 'nn_torch'

train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)
train_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_label.csv', skipinitialspace=True).to_numpy().ravel()
test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True)
test_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_label.csv', skipinitialspace=True).to_numpy().ravel()


train_set1 = torch.Tensor(train_set.values)
train_label1 = torch.LongTensor(train_label)

dataset = MyDataset(train_set1, train_label1)
model = Net(n_features=train_set.shape[1], n_classes=len(np.unique(train_label)))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
batch_size = 32
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

N_EPOCH = 10
for epoch in range(N_EPOCH):
    for batch_features, batch_labels in loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

result = model(torch.Tensor(train_set.values))
pred = result.max(1)[1].numpy()
proba = result.detach().numpy()
print(proba)
print(type(proba))
print(proba.shape)

report = classification_report(train_label, pred, digits=3)
print(report)


model_save_path = f'./models/{MODEL_TYPE}_{DS_NAME}.pt'
model_jit = torch.jit.script(model)
model_jit.save(model_save_path)

model_save_path = f'./models/{MODEL_TYPE}_{DS_NAME}.pkl'
torch.save(model.state_dict(), model_save_path)




"""
train_pred = model.predict(train_set)
train_proba = model.predict_proba(train_set)

test_pred = model.predict(test_set)
test_proba = model.predict_proba(test_set)

print(classification_report(train_label, train_pred))
print(classification_report(test_label, test_pred))
"""