"""
Implementation of the ALOA attack.
"""

import pickle
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler

from AttackModels import AttackThresholdModel
from ._privacy_attack import PrivacyAttack


class AloaPrivacyAttack(PrivacyAttack):

    def __init__(self, black_box,
                 n_shadow_models: int = 1,
                 shadow_model_type: str = 'rf',
                 n_noise_samples_fit: int = 100,
                 n_noise_samples_predict: int | None = None,
                 shadow_test_size: float = 0.5,
                 undersample_attack_dataset: bool = True,
                 percentage_deviation: tuple[float, float] = (0.1, 0.5)):
        super().__init__(black_box, shadow_model_type)
        self.n_shadow_models = n_shadow_models
        self.n_noise_samples_fit = n_noise_samples_fit
        if n_noise_samples_predict is None:
            self.n_noise_samples_predict = n_noise_samples_fit
        else:
            self.n_noise_samples_predict = n_noise_samples_predict
        self.attack_model = None
        self.name = 'aloa_attack'
        self.shadow_test_size = shadow_test_size
        self.undersample_attack_dataset = undersample_attack_dataset
        self.pmin = percentage_deviation[0]
        self.pmax = percentage_deviation[1]

    def fit(self, shadow_dataset: pd.DataFrame, save_files='all', save_folder: str = None):
        if save_folder is None:
            save_folder = f'./{self.name}'
        else:
            save_folder += f'/{self.name}'
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        attack_dataset = self._get_attack_dataset(shadow_dataset, save_files=save_files, save_folder=save_folder)
        class_labels = attack_dataset.pop('class_label')
        target_labels = attack_dataset.pop('target_label')
        scores = self._get_robustness_score(attack_dataset.copy(), class_labels,  self.n_noise_samples_fit)

        # FIXME Should we do a train-test split?
        scores, test_scores, target_labels, test_target_labels = train_test_split(scores, target_labels,
                                                                                  stratify=target_labels, test_size=0.2)

        th_model = AttackThresholdModel()
        th_model.fit(scores, target_labels)
        self.attack_model = th_model
        save_folder += '/attack'
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        # Saving attack model and its performance
        with open(f'{save_folder}/threshold_attack_model_train_performance.txt', 'w', encoding='utf-8') as report:
            report.write(classification_report(target_labels, th_model.predict(scores), digits=3))
            report.write('\n\n')
            report.write(f'Threshold chosen: {th_model.threshold}')
        with open(f'{save_folder}/threshold_attack_model_test_performance.txt', 'w', encoding='utf-8') as report:
            report.write(classification_report(test_target_labels, th_model.predict(test_scores), digits=3))
            report.write('\n\n')
            report.write(f'Threshold chosen: {th_model.threshold}')
        with open(f'{save_folder}/threshold_attack_model.pkl', 'wb') as filename:
            pickle.dump(th_model, filename)
        return th_model.threshold

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        class_labels = self.bb.predict(X)
        scores = self._get_robustness_score(X.copy(), class_labels,  self.n_noise_samples_predict)
        predictions = self.attack_model.predict(scores)
        predictions = np.array(list(map(lambda score: "IN" if score == 1 else "OUT", predictions)))
        return predictions

    def _get_attack_dataset(self, shadow_dataset: pd.DataFrame, save_files='all', save_folder: str = None) -> pd.DataFrame:
        attack_dataset = []
        data_save_folder = save_folder

        if save_files == 'all':
            save_folder += '/shadow'
            Path(save_folder).mkdir(parents=True, exist_ok=True)

        # We audit the black box for the predictions on the shadow set
        labels_shadow = self.bb.predict(shadow_dataset)

        # Train the shadow models
        print(f'Each shadow model uses {max(1/self.n_shadow_models, 0.2)*100:.3f} % of the data')
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

            # Get the "OUT" set
            pred_ts_labels = shadow_model.predict(ts)
            df_out = pd.DataFrame(ts)
            df_out['class_label'] = pred_ts_labels
            df_out['target_label'] = 'OUT'

            if save_files == 'all':
                with open(f'{save_folder}/shadow_model_{self.shadow_model_type}_{i}.pkl', 'wb') as filename:
                    pickle.dump(shadow_model, filename)
                with open(f'{save_folder}/shadow_model_{self.shadow_model_type}_{i}_train_performance.txt', 'w', encoding='utf-8') as report:
                    report.write(classification_report(tr_l, pred_tr_labels, digits=3))
                with open(f'{save_folder}/shadow_model_{self.shadow_model_type}_{i}_test_performance.txt', 'w', encoding='utf-8') as report:
                    report.write(classification_report(ts_l, pred_ts_labels, digits=3))

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
        self.attack_dataset_save_path = f'{data_save_folder}/attack_dataset.csv'
        attack_dataset.to_csv(self.attack_dataset_save_path, index=False)
        return attack_dataset

    def _get_robustness_score(self, dataset, class_labels, n_noise_samples):
        scores = []
        index = 0
        for row in tqdm(dataset.values):
            y_true = class_labels[index]
            y_predicted = self.bb.predict(pd.DataFrame([row]))
            if y_true == y_predicted:
                perturbed_row = row.copy()
                variations = self._generate_noise_neighborhood(perturbed_row, n_noise_samples)
                output = self.bb.predict(pd.DataFrame(variations))
                score = np.mean(np.array(list(map(lambda x: 1 if x == y_true else 0, output))))
                scores.append(score)
            else:
                scores.append(0)
            index += 1
        return scores

    def _generate_noise_neighborhood(self, row, n_noise_samples):
        # Create a matrix by duplicating vect N times
        vect_matrix = np.tile(row, (n_noise_samples, 1))

        # Create a matrix of percentage perturbations to be applied to vect_matrix
        sampl = np.random.uniform(low=self.pmin, high=self.pmax, size=(n_noise_samples, len(row)))
        # Vector for adding or subtracking a value
        sum_sub = np.random.choice([-1, 1], size=(n_noise_samples, len(row)))
        # Here we apply the perturbation perturb
        vect_matrix = vect_matrix + (vect_matrix * (sum_sub * sampl))
        return vect_matrix
