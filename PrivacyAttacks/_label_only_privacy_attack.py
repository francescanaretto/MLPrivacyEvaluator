"""
Implementation of the Label-Only attack.
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


class LabelOnlyPrivacyAttack(PrivacyAttack):

    def __init__(self, black_box,
                 n_shadow_models: int = 1,
                 shadow_model_type: str = 'rf',
                 shadow_model_params: dict = None,
                 n_noise_samples_fit: int = 1000,
                 n_noise_samples_predict: int = None,
                 shadow_test_size: float = 0.5,
                 undersample_attack_dataset: bool = True,
                 prob_bit_flip: float = 0.6):
        super().__init__(black_box, shadow_model_type, shadow_model_params)
        self.n_shadow_models = n_shadow_models
        self.n_noise_samples_fit = n_noise_samples_fit
        if n_noise_samples_predict is None:
            self.n_noise_samples_predict = n_noise_samples_fit
        else:
            self.n_noise_samples_predict = n_noise_samples_predict
        self.attack_model = None
        self.name = 'label_only_attack'
        self.shadow_test_size = shadow_test_size
        self.undersample_attack_dataset = undersample_attack_dataset
        self.prob_bit_flip = prob_bit_flip

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

    def predict(self, X: pd.DataFrame):
        class_labels = self.bb.predict(X)
        scores = self._get_robustness_score(X.copy(), class_labels,  self.n_noise_samples_predict)
        predictions = self.attack_model.predict(scores)
        # predictions = np.array(list(map(lambda score: "IN" if score == 1 else "OUT", predictions)))
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
            attack_dataset, y = undersampler.fit_resample(attack_dataset, y)
            attack_dataset['target_label'] = y
        self.attack_dataset_save_path = f'{data_save_folder}/attack_dataset.csv'
        attack_dataset.to_csv(self.attack_dataset_save_path, index=False)
        return attack_dataset

    def _get_robustness_score(self, dataset, class_labels, n_noise_samples):
        bin_idx, cont_idx = self._get_binary_continuous_features(dataset)
        stdevs = np.array(dataset.std(axis=0))

        scores = []
        index = 0
        for row in tqdm(dataset.values):
            y_true = class_labels[index]
            y_predicted = self.bb.predict(pd.DataFrame([row]))
            if y_true == y_predicted:
                perturbed_row = row.copy()
                variations = self._generate_perturbed_records(perturbed_row, bin_idx, cont_idx, n_noise_samples, stdevs)
                output = self.bb.predict(pd.DataFrame(variations))
                score = np.mean(np.array(list(map(lambda x: 1 if x == y_true else 0, output))))
                scores.append(score)
            else:
                scores.append(0)
            index += 1
        return scores

    def _generate_perturbed_records(self, row, bin_idx, cont_idx,  n_noise_samples, stdevs):
        cont_part = row[cont_idx]
        bin_part = row[bin_idx]

        x_sampled = np.tile(np.copy(row), (n_noise_samples, 1))

        bits_to_flip = np.random.binomial(1, self.prob_bit_flip, (n_noise_samples, len(bin_part)))
        x_flipped = np.invert(bin_part.astype(bool), out=np.copy(x_sampled[:, bin_idx]),
                              where=bits_to_flip.astype(bool)).astype(np.float64)

        noise = stdevs[cont_idx] * np.random.randn(n_noise_samples, len(cont_part))
        x_noisy = x_sampled[:, cont_idx] + noise

        idx = list(np.concatenate([bin_idx, cont_idx]))
        final_idx = [idx.index(x) for x in range(len(idx))]

        x_sampled = np.concatenate([x_flipped, x_noisy], axis=1)
        x_sampled = x_sampled[:, final_idx]
        return x_sampled

    def _get_binary_continuous_features(self, X: pd.DataFrame):
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
