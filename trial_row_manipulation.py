"""
File to test dataset splitting in attack dataset generation.
"""

import pandas as pd
import numpy as np


def _get_binary_continuous_features(X: pd.DataFrame):
    """
    Returns the indexes of columns containing only binary features and only continuous features.
    """
    binary_indices = []
    continuous_indices = []
    for i, column in enumerate(X):
        unique_values = set(X[column].unique())
        if unique_values == {0, 1}:
            binary_indices.append(i)
        else:
            continuous_indices.append(i)
    return binary_indices, continuous_indices


DS_NAME = 'adult'
DATA_FOLDER = f'./data/{DS_NAME}'

train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)
train_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_label.csv', skipinitialspace=True).to_numpy().ravel()
test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True)
test_label = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_label.csv', skipinitialspace=True).to_numpy().ravel()
shadow_data = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_shadow_set.csv', skipinitialspace=True)

bin_idx, cont_idx = _get_binary_continuous_features(train_set)
print(cont_idx)
print(bin_idx)

prob_bit_flip = 0.6
stdevs = np.array(train_set.std(axis=0))

print(type(stdevs))
print(stdevs.shape)
print(stdevs)

n_noise_samples = 40

for row in train_set.values:
    cont_part = row[cont_idx]  # 0 1 3 5
    bin_part = row[bin_idx]  # 2 4
    # concatenated 2 4 0 1 3 5
    # where to go  get_index
    # required     2 3 0 4 1 5
    x_sampled = np.tile(np.copy(row), (n_noise_samples, 1))

    bits_to_flip = np.random.binomial(1, prob_bit_flip, (n_noise_samples, len(bin_part)))

    x_flipped = np.invert(bin_part.astype(bool), out=np.copy(x_sampled[:, bin_idx]),
                          where=bits_to_flip.astype(bool)).astype(np.float64)

    noise = stdevs[cont_idx] * np.random.randn(n_noise_samples, len(cont_part))
    x_noisy = x_sampled[:, cont_idx] + noise

    idx = list(np.concatenate([bin_idx, cont_idx]))
    final_idx = [idx.index(x) for x in range(len(idx))]

    x_sampled = np.concatenate([x_flipped, x_noisy], axis=1)
    x_sampled = x_sampled[:, final_idx]

    break
