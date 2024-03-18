"""
Script to preprocess the adult datset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DS_NAME = "adult"

filename = f"./data/{DS_NAME}/{DS_NAME}_raw.csv"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
df = pd.read_csv(filename, skipinitialspace = True, usecols = columns)

# Droppping duplicates
df.drop_duplicates()

# Print number of NULL values for each column, if any
nulls = dict()
for attr in columns:
    n_null = (df[attr] == '?').sum()
    if n_null:
        nulls[attr] = n_null
print("Number of NUll values: ", nulls)

# Eliminate NULL values
for attr in nulls.keys():
    df.drop(df.index[df[attr]  == '?'], inplace = True)

# Rename class attribute
df.rename(columns = {"salary": "class"}, inplace = True)

# Binarisation of class attribute
df["class"] = df["class"].apply(lambda x: 0 if x == "<=50K" else 1)




# One-hot encoding of categorical attributes
categorical_classes = df.select_dtypes(include = ["object"]).columns.tolist()
df = pd.get_dummies(df, columns = categorical_classes)
print(df.head())
label = df.pop("class")
df.pop('sex_Female')
df.pop('marital-status_Married-civ-spouse')
df.pop('race_Black')

correlation_matrix = df.corr()
print(correlation_matrix.shape)

# Set a correlation threshold (e.g., 0.8)
threshold = 0.70

# Find pairs of highly correlated variables
correlated_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            correlated_pairs.append((colname, correlation_matrix.index[j], correlation_matrix.iloc[i, j]))

# Display highly correlated pairs
for pair in correlated_pairs:
    print(f"Highly correlated variables: {pair[0]} and {pair[1]} with correlation {pair[2]}")


train_set, shadow_set, train_label, shadow_label = train_test_split(df, label, stratify=label,
                                                                    train_size = 0.80, random_state = 14)

# Save original + shadow set
train_set_class = train_label.values
shadow_set_class = shadow_label.values

shadow_set.to_csv(f'./data/{DS_NAME}/{DS_NAME}_shadow_set.csv', index = False)
shadow_label.to_csv(f'./data/{DS_NAME}/{DS_NAME}_shadow_label.csv', index = False)

# Data for model training
train_set, test_set, train_label, test_label = train_test_split(train_set, train_label, stratify = train_label,
                                                                train_size = 0.8, random_state = 15)

# Save dataset
train_set.to_csv(f'./data/{DS_NAME}/{DS_NAME}_original_train_set.csv', index = False)
test_set.to_csv(f'./data/{DS_NAME}/{DS_NAME}_original_test_set.csv', index = False)
train_label.to_csv(f'./data/{DS_NAME}/{DS_NAME}_original_train_label.csv', index = False)
test_label.to_csv(f'./data/{DS_NAME}/{DS_NAME}_original_test_label.csv', index = False)

print("Dataset loaded.")
