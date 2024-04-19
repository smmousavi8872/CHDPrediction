from collections import Counter

from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
import pandas as pd


class OverSample:

    def augment_dataset(self, ip_data):
        from sklearn.utils import shuffle
        X = ip_data.iloc[:, :-1]
        y = ip_data.CoronaryHeartDisease
        over_sample = SMOTE()
        smote_X, smote_y = over_sample.fit_resample(X, y)
        smote_X = pd.DataFrame(smote_X)
        smote_y = pd.DataFrame(smote_y)
        aug_data = pd.concat([smote_X, smote_y], axis=1)
        aug_data = shuffle(aug_data)
        CHD_label = aug_data.CoronaryHeartDisease
        counter = Counter(CHD_label)
        print("shape of augmented dataset" + str(aug_data.shape))
        print("distribution of augmented dataset = " + str(counter))
        return aug_data

    def random(self, x_set, y, rand_state):
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=rand_state, sampling_strategy='minority')
        X_resampled, y_resampled = ros.fit_resample(x_set, y)
        return X_resampled, y_resampled

    def adasyn(self, x_set, y):
        from imblearn.over_sampling import ADASYN
        X_resampled, y_resampled = ADASYN(sampling_strategy='minority').fit_resample(x_set, y)
        return X_resampled, y_resampled

    def smote(self, x_set, y):
        X_resampled, y_resampled = SMOTE(sampling_strategy='minority').fit_resample(x_set, y)
        return X_resampled, y_resampled

    def borderline_smote(self, x_set, y):
        X_resampled, y_resampled = BorderlineSMOTE(sampling_strategy='minority').fit_resample(x_set, y)
        return X_resampled, y_resampled

    def k_means_smote(self, x_set, y):
        # Make this a binary classification problem
        y = y == 1
        X_resampled, y_resampled = KMeansSMOTE(sampling_strategy='minority', random_state=42).fit_resample(x_set, y)
        return X_resampled, y_resampled

    def svm_smote(self, x_set, y):
        X_resampled, y_resampled = SVMSMOTE(sampling_strategy='minority').fit_resample(x_set, y)
        return X_resampled, y_resampled
