

from tqdm import tqdm
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler

from ShadowModels import ShadowRandomForest
from AttackModels import AttackThresholdModel
from ._privacy_attack import PrivacyAttack


class AloaPrivacyAttack(PrivacyAttack):

    def __init__(self, black_box, n_shadow_models='1', shadow_model_type='rf',
                 n_noise_samples_fit=100, n_noise_samples_predict=None,
                 shadow_test_size=0.5, undersample_attack_dataset=True):
        super().__init__(black_box)
        self.n_shadow_models = n_shadow_models
        self.shadow_model_type = shadow_model_type
        self.n_noise_samples_fit = n_noise_samples_fit
        if n_noise_samples_predict is None:
            self.n_noise_samples_predict = n_noise_samples_fit
        else:
            self.n_noise_samples_predict = n_noise_samples_predict
        self.attack_model = None
        self.name = 'aloa_attack'
        self.shadow_test_size = shadow_test_size
        self.undersample_attack_dataset = undersample_attack_dataset

    def fit(self, shadow_dataset: pd.DataFrame, save_files='all', save_folder: str = None):
        if save_folder is None:
            save_folder = f'./{self.name}'
        else:
            save_folder += f'/{self.name}'
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        attack_dataset = self._get_attack_dataset(shadow_dataset)
        class_labels = attack_dataset.pop('class_label')
        target_labels = attack_dataset.pop('target_label')
        scores = self._get_robustness_score(attack_dataset.copy(), class_labels,  self.n_noise_samples_fit)
        # Convert IN/OUT to 1/0 for training the threshold model
        target_labels = np.array(list(map(lambda score: 0 if score == "OUT" else 1, target_labels)))
        th_model = AttackThresholdModel()
        th_model.fit(scores, target_labels)
        self.attack_model = th_model
        return th_model.threshold

    def predict(self, X: pd.DataFrame):
        class_labels = self.bb.predict(X)
        scores = self._get_robustness_score(X.copy(), class_labels,  self.n_noise_samples_predict)
        predictions = self.attack_model.predict(scores)
        predictions = np.array(list(map(lambda score: "IN" if score == 1 else "OUT", predictions)))
        return predictions

    def _get_attack_dataset(self, shadow_dataset: pd.DataFrame):
        attack_dataset = []
        # We audit the black box for the predictions on the shadow set
        labels_shadow = self.bb.predict(shadow_dataset)

        # Train the shadow models
        for i in range(1, self.n_shadow_models+1):
            data = shadow_dataset.sample(frac=max(1/self.n_shadow_models, 0.2), replace=False)
            labels = labels_shadow[np.array(data.index)]

            tr, ts, tr_l, ts_l = train_test_split(data, labels, stratify=labels, test_size=self.shadow_test_size)

            # Create and train the shadow model
            shadow_model = self._get_shadow_model()
            shadow_model.fit(tr, tr_l)

            # Get the "IN" set
            pred_tr_labels = shadow_model.predict(tr)
            df_in = pd.DataFrame(tr)
            df_in['class_label'] = pred_tr_labels
            df_in['target_label'] = 'IN'
            # print(classification_report(tr_l, pred_tr_labels, digits=3))

            # Get the "OUT" set
            pred_ts_labels = shadow_model.predict(ts)
            df_out = pd.DataFrame(ts)
            df_out['class_label'] = pred_ts_labels
            df_out['target_label'] = 'OUT'
            # print(classification_report(ts_l, pred_ts_labels, digits=3))

            df_final = pd.concat([df_in, df_out])
            attack_dataset.append(df_final)

        # Merge all sets and reset the index
        attack_dataset = pd.concat(attack_dataset)
        attack_dataset = attack_dataset.reset_index(drop=True)
        if self.undersample_attack_dataset:
            undersampler = RandomUnderSampler(sampling_strategy='majority')
            y = attack_dataset['target_label']
            attack_dataset.columns = attack_dataset.columns.astype(str)
            attack_dataset, _ = undersampler.fit_resample(attack_dataset, y)
        attack_dataset.to_csv('./data/attack_dataset_aloa.csv', index=False)  # DO WE SAVE THE ATTACK DATASET?
        return attack_dataset

    def _get_robustness_score(self, dataset, class_labels, n_noise_samples):
        percentage_deviation = (0.1, 0.50)
        scores = []
        index = 0
        for row in tqdm(dataset.values):
            variations = []
            y_true = class_labels[index]
            y_predicted = self.bb.predict(np.array([row]))
            if y_true == y_predicted:
                perturbed_row = row.copy()
                variations = self._noise_neighborhood(perturbed_row, n_noise_samples, percentage_deviation)
                output = self.bb.predict(variations)
                score = np.mean(np.array(list(map(lambda x: 1 if x == y_true else 0, output))))
                scores.append(score)
            else:
                scores.append(0)
            index += 1
        return scores

    def _noise_neighborhood(self, row, n_noise_samples, percentage_deviation):
        pmin = percentage_deviation[0]
        pmax = percentage_deviation[1]
        # Create a matrix by duplicating vect N times
        vect_matrix = np.tile(row, (n_noise_samples, 1))

        # Create a matrix of percentage perturbations to be applied to vect_matrix
        sampl = np.random.uniform(low=pmin, high=pmax, size=(n_noise_samples, len(row)))
        # Vector for adding or subtracking a value
        sum_sub = np.random.choice([-1, 1], size=(n_noise_samples, len(row)))
        # Here we apply the perturbation perturb
        vect_matrix = vect_matrix + (vect_matrix * (sum_sub * sampl))
        return vect_matrix

    def _get_shadow_model(self):
        if self.shadow_model_type == 'rf':
            shadow_model = ShadowRandomForest()
        return shadow_model
