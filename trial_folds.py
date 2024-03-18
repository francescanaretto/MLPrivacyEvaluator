
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupKFold, KFold

ds_name = 'adult'

train_data = pd.read_csv(f'./data/{ds_name}/{ds_name}_original_train_set.csv', skipinitialspace = True)
train_label = pd.read_csv(f'./data/{ds_name}/{ds_name}_original_train_label.csv', skipinitialspace = True).to_numpy().ravel()
test_data = pd.read_csv(f'./data/{ds_name}/{ds_name}_original_test_set.csv', skipinitialspace = True)
test_label = pd.read_csv(f'./data/{ds_name}/{ds_name}_original_test_label.csv', skipinitialspace = True).to_numpy().ravel()
shadow_data = pd.read_csv(f'./data/{ds_name}/{ds_name}_shadow_set.csv', skipinitialspace = True)

folder = KFold(n_splits=3)

print(shadow_data.shape)

for y, x in folder.split(shadow_data):
    data_train = shadow_data.iloc[x]
    print(type(data_train))


strat = StratifiedKFold(n_splits=3)
for y, x in strat.split(test_data, test_label):
    print(x.shape)
    print(y.shape)
    print(type(y))
    print(test_label[y])
