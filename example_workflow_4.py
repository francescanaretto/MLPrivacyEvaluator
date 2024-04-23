import warnings

import pandas as pd

from MLWrappers import SklearnBlackBox, PyTorchBlackBox, KerasBlackBox
from PrivacyAttacks import MiaPrivacyAttack, LabelOnlyPrivacyAttack, AloaPrivacyAttack
from MLPrivacyEvaluator import PrivacyEvaluator

warnings.simplefilter("ignore", UserWarning)


DS_NAME = 'adult'
DATA_FOLDER = f'./data/{DS_NAME}'

# we load the target black box model using our wrapper
# target = SklearnBlackBox(f'./models/{DS_NAME}_dt.pkl')
target = SklearnBlackBox(f'./models/{DS_NAME}_rf.pkl')
# target = PyTorchBlackBox(f'./models/{DS_NAME}_nn_torch.pt')
# target = KerasBlackBox(f'./models/{DS_NAME}_nn_keras.keras')

# We load the data used to train, test of the model, as well as the shadow data
train_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_train_set.csv', skipinitialspace=True)
test_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_original_test_set.csv', skipinitialspace=True)
shadow_set = pd.read_csv(f'{DATA_FOLDER}/{DS_NAME}_shadow_set.csv', skipinitialspace=True)

shadow_tree_params = {'min_samples_leaf': 100,
                      'max_depth': 10}
shadow_forest_params = {'n_estimators': 100,
                        'n_jobs': 8}

# We initialise the attacks, with the desired parameters for each
mia = MiaPrivacyAttack(target,
                       n_shadow_models=3)
label_only = LabelOnlyPrivacyAttack(target,
                                    n_shadow_models=1,
                                    shadow_model_type='rf',
                                    shadow_model_params=shadow_forest_params,
                                    n_noise_samples_fit=1000)
aloa = AloaPrivacyAttack(target,
                         n_shadow_models=1,
                         shadow_model_type='rf',
                         shadow_model_params=shadow_forest_params,
                         n_noise_samples_fit=1000,
                         shadow_test_size=0.3,
                         undersample_attack_dataset=True)
attacks = [mia, label_only, aloa]

# We initialise the PrivacyEvaluator object
# We pass the target model and the attacks we want to use
evaluator = PrivacyEvaluator(target, attacks)

# We use the fit() method to execute the attacks, starting from the shadow data
evaluator.fit(shadow_set)

# Then we can obtain the performances using the report() method
results = evaluator.report(train_set, test_set)
print(results)

print(results['mia_attack']['classification_report'])
print(results['label_only_attack']['classification_report'])
print(results['aloa_attack']['classification_report'])
