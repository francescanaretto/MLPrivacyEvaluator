
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, classification_report
from AttackModels import AttackThresholdModel

data = pd.read_csv('./data/attack_dataset.csv', skipinitialspace=True)

train_set = pd.DataFrame(data['0']).to_numpy().ravel()
train_label = pd.DataFrame(data['class_label']).to_numpy().ravel()


model = AttackThresholdModel()
model.fit(train_set, train_label, thresholds=np.linspace(0,1,100), score_type='accuracy')

pred = model.predict(train_set)

print(pred)

test_set = pd.DataFrame(data['1']).to_numpy().ravel()
print(classification_report(train_label, model.predict(test_set)))
