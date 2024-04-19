"""
Example use of the MLPrivacyEvaluator library.
"""

import warnings

import pandas as pd
import torch

from MLWrappers import SklearnBlackBox, PyTorchBlackBox
from PrivacyAttacks import MiaPrivacyAttack, AloaPrivacyAttack
from MLPrivacyEvaluator import PrivacyEvaluator


warnings.simplefilter("ignore", UserWarning)


class Net(torch.nn.Module):
    
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        
        self.fc1 = torch.nn.Linear(self.n_features, 64)
        self.fc99 = torch.nn.Linear(64, self.n_classes)
        
        self.dropout = torch.nn.Dropout(0.30)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        out = torch.nn.functional.softmax(self.fc99(x), dim = 1)
        return out


DS_NAME = 'adult'
DATA_FOLDER = f'./data/{DS_NAME}_thesis'

# we load the target black box model using our wrapper
# target = SklearnBlackBox(f'./models/rf_{DS_NAME}.sav')
target = PyTorchBlackBox(f'./models/nn_torch_{DS_NAME}_thesis.sav', nn_class=Net(104, 2))

# We load the data used to train, test of the model, as well as the shadow data
train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)
test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True)
shadow_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_stat_shadow.csv', skipinitialspace=True)

# We initialise the attacks, with the desired parameters for each
mia = MiaPrivacyAttack(target, n_shadow_models=3)
aloa = AloaPrivacyAttack(target, n_shadow_models=1, n_noise_samples_fit=1000,
                         shadow_test_size=0.2, undersample_attack_dataset=True)
attacks = [mia, aloa]

# We initialise the PrivacyEvaluator object
# We pass the target model and the attacks we want to use
evaluator = PrivacyEvaluator(target, attacks)

# We use the fit() method to execute the attacks, starting from the shadow data
evaluator.fit(shadow_set, save_folder='./new_save_folder')

# Then we can obtain the performances using the report() method
results = evaluator.report(train_set, test_set)
print(results)
