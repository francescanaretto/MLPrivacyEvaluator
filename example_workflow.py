"""
Example use of the MLPrivacyEvaluator library.
"""

import warnings

import pandas as pd

from MLWrappers import SklearnBlackBox
from PrivacyAttacks import MiaPrivacyAttack, AloaPrivacyAttack
from MLPrivacyEvaluator import PrivacyEvaluator


warnings.simplefilter("ignore", UserWarning)

DS_NAME = 'gaussian'
DATA_FOLDER = f'./data/{DS_NAME}'

# we load the target black box model using our wrapper
target = SklearnBlackBox(f'./models/rf_{DS_NAME}.sav')

# We load the data used to train, test of the model, as well as the shadow data
train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)[:800]
test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True)[:400]
shadow_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_shadow_set.csv', skipinitialspace=True)[:500]

# We initialise the attacks, with the desired parameters for each
mia = MiaPrivacyAttack(target, n_shadow_models=5)
aloa = AloaPrivacyAttack(target, n_shadow_models=1, n_noise_samples_fit=200)
attacks = [mia, aloa]

# We initialise the PrivacyEvaluator object
# We pass the target model and the attacks we want to use
evaluator = PrivacyEvaluator(target, attacks)

# We use the fit() method to execute the attacks, starting from the shadow data
evaluator.fit(shadow_set)

# Then we can obtain the performances using the report() method
results = evaluator.report(train_set, test_set)
print(results)
