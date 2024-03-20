
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from PrivacyAttacks.privacy_attack import PrivacyAttack
from ShadowModels.random_forest_shadow_model import ShadowRandomForest
from AttackModels.threshold_attack_model import AttackThresholdModel


class AloaPrivacyAttack(PrivacyAttack):
    def __init__(self, black_box, n_shadow_models='1', shadow_model_type = 'rf'):
        super().__init__(black_box)
        self.n_shadow_models = n_shadow_models
        self.shadow_model_type = shadow_model_type
        self.attack_model = None

    def fit(self, shadow_dataset: pd.DataFrame, n_noise_samples=100, attack_model_path: str = './attack_models'):
        attack_dataset = self._get_attack_dataset(shadow_dataset)
        class_labels = attack_dataset.pop('class_label')
        target_labels = attack_dataset.pop('target_label')
        scores = self._get_robustness_score(attack_dataset.copy(), class_labels,  n_noise_samples)
        # Convert IN/OUT to 1/0 for training the threshold model
        target_labels = np.array(list(map(lambda score:0 if score == "OUT" else 1, target_labels)))
        th_model = AttackThresholdModel()
        th_model.fit(scores, target_labels)
        self.attack_model = th_model
        return th_model.threshold

    def predict(self, X: pd.DataFrame, n_noise_samples=100):
        class_labels = self.bb.predict(X)
        scores = self._get_robustness_score(X.copy(), class_labels,  n_noise_samples)
        predictions = self.attack_model.predict(scores)
        predictions = np.array(list(map(lambda score:"IN" if score == 1 else "OUT", predictions)))
        return predictions

    def _get_attack_dataset(self, shadow_dataset: pd.DataFrame):
        attack_dataset = []
        # We audit the black box for the predictions on the shadow set
        labels_shadow = self.bb.predict(shadow_dataset)

                # Train the shadow models
        if self.n_shadow_models >= 2:
            folds = StratifiedKFold(n_splits=self.n_shadow_models)
            # tr and ts are inverted when selecting the fold
            for _, (ts_index, tr_index) in enumerate(folds.split(shadow_dataset, labels_shadow)):
                # Get train and test data for each shadow model
                tr = shadow_dataset.iloc[tr_index]
                tr_l = labels_shadow[tr_index]
                ts = shadow_dataset.iloc[ts_index]
                ts_l = labels_shadow[ts_index]

                # Create and train the shadow model
                shadow_model = self._get_shadow_model()
                shadow_model.fit(tr, tr_l)

                # Get the "IN" set
                pred_tr_labels = shadow_model.predict(tr)
                df_in = pd.DataFrame(tr)
                df_in['class_label'] = pred_tr_labels
                df_in['target_label'] = 'IN'
                #print(classification_report(tr_l, pred_tr_labels, digits=3))

                # Get the "OUT" set
                pred_ts_labels = shadow_model.predict(ts)
                df_out = pd.DataFrame(ts)
                df_out['class_label'] = pred_ts_labels
                df_out['target_label'] = 'OUT'
                #print(classification_report(ts_l, pred_ts_labels, digits=3))

                df_final = pd.concat([df_in, df_out])
                attack_dataset.append(df_final)
        else:
            tr, ts, tr_l, ts_l = train_test_split(shadow_dataset, labels_shadow, stratify=labels_shadow, test_size=0.2)

            # Create and train the shadow model
            shadow_model = self._get_shadow_model()
            shadow_model.fit(tr, tr_l)

            # Get the "IN" set
            pred_tr_labels = shadow_model.predict(tr)
            df_in = pd.DataFrame(tr)
            df_in['class_label'] = pred_tr_labels
            df_in['target_label'] = 'IN'
            #print(classification_report(tr_l, pred_tr_labels, digits=3))

            # Get the "OUT" set
            pred_ts_labels = shadow_model.predict(ts)
            df_out = pd.DataFrame(ts)
            df_out['class_label'] = pred_ts_labels
            df_out['target_label'] = 'OUT'
            #print(classification_report(ts_l, pred_ts_labels, digits=3))

            df_final = pd.concat([df_in, df_out])
            attack_dataset.append(df_final)

        # Merge all sets and reset the index
        attack_dataset = pd.concat(attack_dataset)
        attack_dataset = attack_dataset.reset_index(drop=True)
        undersampler = RandomUnderSampler(sampling_strategy='majority')
        y = attack_dataset['target_label']
        attack_dataset.columns = attack_dataset.columns.astype(str)
        attack_dataset, _= undersampler.fit_resample(attack_dataset, y)
        attack_dataset.to_csv('./data/attack_dataset_aloa.csv', index=False) # DO WE SAVE THE ATTACK DATASET?
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
