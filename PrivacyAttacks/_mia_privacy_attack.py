"""
Implementation of the original MIA attack.
"""

import pickle
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler

from MLWrappers._bbox import AbstractBBox
from ._privacy_attack import PrivacyAttack


class MiaPrivacyAttack(PrivacyAttack):
    """Class implementing the original MIA attack by Shokri et al. (2017)."""

    def __init__(self, black_box: AbstractBBox,
                 name: str = 'mia_attack',
                 n_shadow_models: int = 3,
                 shadow_model_type: str = 'rf',
                 shadow_model_params: dict = None,
                 attack_model_type: str = 'rf',
                 attack_model_params: dict = None,
                 shadow_test_size: float = 0.5,
                 undersample_shadow_dataset: bool = False,
                 undersample_attack_dataset: bool = True,
                 voting_model: bool = False):
        """
        Class implementing the original MIA attack by Shokri et al. (2017).

        Parameters
        ----------
        black_box : AbstractBBox
            The target machine learning model to be attacked.
        name : str, default='mia_attack'
            Name to be used for file saving.
        n_shadow_models : int, default=3
            Number of shadow models to be used.
        shadow_model_type : str, {'rf', 'dt'}, default='rf'
            Type of shadow model to be used.
        shadow_model_params : dict, optional
            Parameters to be passed to the shadow model.
        attack_model_type : str, {'rf', 'dt'}, default='rf'
            Type of attack model to be used.
        attack_model_params : dict, optional
            Parameters to be passed to the attack model.
        shadow_test_size : float, default=0.5
            test set size used during shadow model training.
        undersample_attack_dataset : bool, default=False
            Whether to balance the shadow dataset or not. If True, it will undersample the black box predictions on the
            shadow set to have balanced classes.
        undersample_attack_dataset : bool, default=True
            Whether to balance the attack dataset or not. If True, it will undersample the majority class to obtain
            balanced IN/OUT classes.
        voting_model : bool, default=False
            Whether to use a voting attack model or not. If True, the prediction of the attack will be a majority vote
            of all attack models. If False, the prediciton of the attack will be the prediction of the attack model
            corresponding to the class of the given sample.
        """
        super().__init__(black_box, shadow_model_type, shadow_model_params, attack_model_type, attack_model_params)
        self.name = name
        self.n_shadow_models = n_shadow_models
        self.shadow_test_size = shadow_test_size
        self.undersample_shadow_dataset = undersample_shadow_dataset
        self.undersample_attack_dataset = undersample_attack_dataset
        self.voting_model = voting_model
        self.attack_models = None
        self.attack_dataset_save_path = None

    def fit(self, shadow_dataset: pd.DataFrame, save_files: str = 'all', save_folder: str = None):
        if save_folder is None:
            save_folder = f'./{self.name}'
        else:
            save_folder += f'/{self.name}'
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        attack_dataset = self._get_attack_dataset(shadow_dataset, save_files=save_files, save_folder=save_folder)

        # Obtain list of all class labels
        classes = list(attack_dataset['class_label'].unique())
        self.attack_models = [None] * len(classes)

        save_folder += '/attack'
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        # Train one model for each class
        for c in classes:
            tr = attack_dataset[attack_dataset['class_label'] == c]  # Select only records of that class
            tr.pop('class_label')  # Drop class attribute
            tr_l = np.array(tr.pop('target_label'))  # Use IN/OUT as labels

            attack_model = self._get_attack_model()

            train_set, test_set, train_label, test_label = train_test_split(tr, tr_l, stratify=tr_l, test_size=0.2)
            attack_model.fit(train_set.values, train_label)
            with open(f'{save_folder}/attack_model_{self.attack_model_type}_class_{c}_train_performance.txt', 'w',
                      encoding='utf-8') as report:
                report.write(classification_report(train_label, attack_model.predict(train_set), digits=3))
            with open(f'{save_folder}/attack_model_{self.attack_model_type}_class_{c}_test_performance.txt', 'w',
                      encoding='utf-8') as report:
                report.write(classification_report(test_label, attack_model.predict(test_set), digits=3))

            with open(f'{save_folder}/attack_model_{self.attack_model_type}_class_{c}.pkl', 'wb') as filename:
                pickle.dump(attack_model, filename)

            self.attack_models[c] = attack_model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        class_labels = self.bb.predict(X)
        proba = pd.DataFrame(self.bb.predict_proba(X))
        class_labels = np.argmax(self.bb.predict_proba(X), axis=1)
        predictions = []
        if self.voting_model:
            # TODO implement voting attack model
            pass
        else:
            for idx, row in enumerate(tqdm(proba.values)):
                # pred = self.attack_models[class_labels[idx]].predict(row.reshape(1, -1))
                pred = self.attack_models[class_labels[idx]].predict(pd.DataFrame(row.reshape(1, -1)))
                predictions.extend(pred)
        return np.array(predictions)

    def _get_attack_dataset(self, shadow_dataset: pd.DataFrame, save_files='all', save_folder: str = None):
        attack_dataset = []
        data_save_folder = save_folder

        if save_files == 'all':
            save_folder += '/shadow'
            Path(save_folder).mkdir(parents=True, exist_ok=True)

        # We audit the black box for the predictions on the shadow set
        labels_shadow = self.bb.predict(shadow_dataset)
        if self.undersample_shadow_dataset:
            undersampler = RandomUnderSampler(sampling_strategy='majority')
            shadow_dataset.columns = shadow_dataset.columns.astype(str)
            shadow_dataset, labels_shadow = undersampler.fit_resample(shadow_dataset, labels_shadow)
            shadow_dataset = shadow_dataset.reset_index(drop=True)

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
            pred_tr_proba = shadow_model.predict_proba(tr)
            df_in = pd.DataFrame(pred_tr_proba)
            df_in['class_label'] = pred_tr_labels
            df_in['target_label'] = 'IN'

            # Get the "OUT" set
            pred_ts_labels = shadow_model.predict(ts)
            pred_ts_proba = shadow_model.predict_proba(ts)
            df_out = pd.DataFrame(pred_ts_proba)
            df_out['class_label'] = pred_ts_labels
            df_out['target_label'] = 'OUT'

            if save_files == 'all':
                with open(f'{save_folder}/shadow_model_{self.shadow_model_type}_{i}.pkl', 'wb') as filename:
                    pickle.dump(shadow_model, filename)
                with open(f'{save_folder}/shadow_model_{self.shadow_model_type}_{i}_train_performance.txt', 'w',
                          encoding='utf-8') as report:
                    report.write(classification_report(tr_l, pred_tr_labels, digits=3))
                with open(f'{save_folder}/shadow_model_{self.shadow_model_type}_{i}_test_performance.txt', 'w',
                          encoding='utf-8') as report:
                    report.write(classification_report(ts_l, pred_ts_labels, digits=3))

            df_final = pd.concat([df_in, df_out])
            attack_dataset.append(df_final)

        # Merge all sets and reset the index
        attack_dataset = pd.concat(attack_dataset)
        attack_dataset = attack_dataset.reset_index(drop=True)
        if self.undersample_attack_dataset:
            undersampler = RandomUnderSampler(sampling_strategy='majority')
            attack_dataset.columns = attack_dataset.columns.astype(str)
            attack_dataset, _ = undersampler.fit_resample(attack_dataset, attack_dataset['target_label'])
            attack_dataset = attack_dataset.reset_index(drop=True)
        self.attack_dataset_save_path = f'{data_save_folder}/attack_dataset.csv'
        attack_dataset.to_csv(self.attack_dataset_save_path, index=False)
        return attack_dataset
