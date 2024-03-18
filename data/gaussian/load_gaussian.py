"""
Script to preprocess and load a synthetic gaussian dataset with two classes.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DS_NAME = 'gaussian'
class_name = 'target'

filename = f'./data/{DS_NAME}/{DS_NAME}_raw.csv'
df = pd.read_csv(filename)

df.drop_duplicates()
df.pop('Unnamed: 0')
df.rename(columns = {'target': 'class'}, inplace = True)
label = df.pop("class")

train_set, shadow_set, train_label, shadow_label = train_test_split(df, label, stratify=label,
                                                                    train_size = 0.80, random_state = 42)


shadow_set.to_csv(f'./data/{DS_NAME}/{DS_NAME}_shadow_set.csv', index = False)
shadow_label.to_csv(f'./data/{DS_NAME}/{DS_NAME}_shadow_label.csv', index = False)

# Data for model training
train_set, test_set, train_label, test_label = train_test_split(train_set, train_label, stratify = train_label,
                                                                train_size = 0.8, random_state = 43)

# Save dataset
train_set.to_csv(f'./data/{DS_NAME}/{DS_NAME}_original_train_set.csv', index = False)
test_set.to_csv(f'./data/{DS_NAME}/{DS_NAME}_original_test_set.csv', index = False)
train_label.to_csv(f'./data/{DS_NAME}/{DS_NAME}_original_train_label.csv', index = False)
test_label.to_csv(f'./data/{DS_NAME}/{DS_NAME}_original_test_label.csv', index = False)

print("Dataset loaded.")
