"""
Implementation of the original MIA attack.
"""

import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler

from MLWrappers._bbox import AbstractBBox
from ShadowModels import ShadowRandomForest
from AttackModels import AttackRandomForest
from ._privacy_attack import PrivacyAttack


class MiaPrivacyAttack(PrivacyAttack):

    def __init__(self, black_box: AbstractBBox, n_shadow_models=3, shadow_model_type='rf', attack_model_type='rf'):
        super().__init__(black_box)
        self.n_shadow_models = n_shadow_models
        self.shadow_model_type = shadow_model_type
        self.attack_model_type = attack_model_type
        self.name = 'mia_attack'

    def fit(self, shadow_dataset: pd.DataFrame, attack_model_path: str = './attack_models'):
        attack_dataset = self._get_attack_dataset(shadow_dataset)
        # Obtain list of all class labels
        classes = list(attack_dataset['class_label'].unique())
        self.attack_models = [None] * len(classes)
        # Train one model for each class
        for c in classes:
            tr = attack_dataset[attack_dataset['class_label'] == c]  # Select only records of that class
            tr.pop('class_label')  # Drop class attribute
            tr_l = tr.pop('target_label')  # Use IN/OUT as labels

            attack_model = self._get_attack_model()

            train_set, test_set, train_label, test_label = train_test_split(tr, tr_l, stratify=tr_l, test_size=0.2)
            attack_model.fit(train_set.values, train_label)
            with open(f'{attack_model_path}/attack_model_{self.attack_model_type}_class_{c}_train_performance.txt', 'w', encoding='utf-8') as report:
                report.write(classification_report(train_label, attack_model.predict(train_set), digits=3))
            with open(f'{attack_model_path}/attack_model_{self.attack_model_type}_class_{c}_test_performance.txt', 'w', encoding='utf-8') as report:
                report.write(classification_report(test_label, attack_model.predict(test_set), digits=3))

            with open(f'{attack_model_path}/attack_model_{self.attack_model_type}_class_{c}.sav', 'wb') as filename:
                pickle.dump(attack_model, filename)

            self.attack_models[c] = attack_model
        return self.attack_models

    def predict(self, X: pd.DataFrame):
        class_labels = self.bb.predict(X)
        proba = pd.DataFrame(self.bb.predict_proba(X))
        class_labels = np.argmax(self.bb.predict_proba(X), axis=1)
        predictions = []
        for idx, row in enumerate(proba.values):
            pred = self.attack_models[class_labels[idx]].predict(row.reshape(1, -1))
            predictions.extend(pred)
        return np.array(predictions)

    def _get_shadow_model(self):
        if self.shadow_model_type == 'rf':
            shadow_model = ShadowRandomForest()
        return shadow_model

    def _get_attack_dataset(self, shadow_dataset: pd.DataFrame):
        attack_dataset = []

        # We audit the black box for the predictions on the shadow set
        labels_shadow = self.bb.predict(shadow_dataset)

        # Train the shadow models
        for i in range(1, self.n_shadow_models+1):
            data = shadow_dataset.sample(frac=max(1/self.n_shadow_models, 0.2), replace=False)
            labels = labels_shadow[np.array(data.index)]

            tr, ts, tr_l, ts_l = train_test_split(data, labels, stratify=labels, test_size=0.5)
            # Create and train the shadow model
            shadow_model = self._get_shadow_model()
            shadow_model.fit(tr, tr_l)

            # Get the "IN" set
            pred_tr_labels = shadow_model.predict(tr)
            pred_tr_proba = shadow_model.predict_proba(tr)
            df_in = pd.DataFrame(pred_tr_proba)
            df_in['class_label'] = pred_tr_labels
            df_in['target_label'] = 'IN'
            # print(classification_report(tr_l, pred_tr_labels, digits=3))

            # Get the "OUT" set
            pred_ts_labels = shadow_model.predict(ts)
            pred_ts_proba = shadow_model.predict_proba(ts)
            df_out = pd.DataFrame(pred_ts_proba)
            df_out['class_label'] = pred_ts_labels
            df_out['target_label'] = 'OUT'
            # print(classification_report(ts_l, pred_ts_labels, digits=3))

            df_final = pd.concat([df_in, df_out])
            attack_dataset.append(df_final)

        # Merge all sets and reset the index
        attack_dataset = pd.concat(attack_dataset)
        attack_dataset = attack_dataset.reset_index(drop=True)
        undersampler = RandomUnderSampler(sampling_strategy='majority')
        y = attack_dataset['target_label']
        attack_dataset.columns = attack_dataset.columns.astype(str)
        attack_dataset, _ = undersampler.fit_resample(attack_dataset, y)
        attack_dataset.to_csv('./data/attack_dataset.csv', index=False)  # DO WE SAVE THE ATTACK DATASET?
        return attack_dataset

    def _get_attack_model(self):
        if self.attack_model_type == 'rf':
            model = AttackRandomForest()
        return model
