import pandas as pd
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'
from imblearn.under_sampling import RandomUnderSampler

FOLDER = f'./data/adult_paper'

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
df = pd.read_csv(f"{FOLDER}/adult_data.csv", skipinitialspace=True, usecols=columns)

# Duplicated drop.
df = df.drop_duplicates()

# Deleting missing values.
df.drop(df.index[df['workclass'] == '?'], inplace=True)
df.drop(df.index[df['occupation'] == '?'], inplace=True)
df.drop(df.index[df['native-country'] == '?'], inplace=True)

# Binarizzation for the feature class salary for predicting purpose.
df.rename(columns={'salary': 'class'}, inplace=True)  # renaming the salary column to class.
df['class'] = df['class'].apply(lambda x: 0 if x == "<=50K" else 1)
categorical_classes = df.select_dtypes(include=['object']).columns.tolist()

# Hot encoding of all the categorical attributes.
df = pd.get_dummies(df, columns=categorical_classes)
label_dt = df.pop('class')

undersample = RandomUnderSampler(sampling_strategy="majority")
tr, tr_l = undersample.fit_resample(df, label_dt)

train_set, shadow_set, train_label, shadow_label = train_test_split(tr, tr_l, stratify=tr_l,
                                                                    test_size=0.80, random_state=1)

# Saving the shadow set and the original dataset.
shadow_set['class'] = shadow_label.values
train_set['class'] = train_label.values
# This set will be used later to train and evaluate the model.
shadow_set[:train_set.shape[0]].to_csv('../../data/adult/adult_shadow.csv', index=False)

train_label = train_set.pop('class')
# Splittig the original dataset train set with the tipical hold out percentage 80-20.
train_set, test_set, train_label, test_label = train_test_split(train_set, train_label, stratify=train_label,
                                                                test_size=0.20, random_state=0)
train_set.to_csv('../../data/adult/original_train_set.csv', index=False)
test_set.to_csv('../../data/adult/original_test_set.csv', index=False)
train_label.to_csv('../../data/adult/original_train_label.csv', index=False)
test_label.to_csv('../../data/adult/original_test_label.csv', index=False)