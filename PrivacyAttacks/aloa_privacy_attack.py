
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from PrivacyAttacks.privacy_attack import PrivacyAttack
from ShadowModels.random_forest_shadow_model import ShadowRandomForest


class AloaPrivacyAttack(PrivacyAttack):
    def __init__(self, black_box, n_shadow_models='1', shadow_model_type = 'rf'):
        super().__init__(black_box)
        self.n_shadow_models = n_shadow_models
        self.shadow_model_type = shadow_model_type

    def _get_binary_continuous_features(self, X: pd.DataFrame):
        """
        Returns the indexes of columns containing only binary features and only continuous features.
        """
        binary_indices = []
        continuous_indices = []
        for i, column in enumerate(X):
            unique_values = set(X[column].unique())
            if unique_values == set([0, 1]):
                binary_indices.append(i)
            else:
                continuous_indices.append(i)
        return binary_indices, continuous_indices
    
    def _continuous_noise(self):
        pass

    def _binary_flip(self):
        pass

    def _get_shadow_model(self):
        if self.shadow_model_type == 'rf':
            shadow_model = ShadowRandomForest()
        return shadow_model

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

    def fit(self, shadow_dataset: pd.DataFrame):
        pass

    def predict(self):
        pass
