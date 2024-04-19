import statistics
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from matplotlib import pyplot as plt
from numpy.random import seed
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.manifold import TSNE  # t-SNE visualization
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras

from OverSample import OverSample

np.random.seed(2095)


class CHDPredictionCV:
    global varb  # selected features by LASSO
    global featureIndex  # indexes of selected features by LASSO
    global feature_weights  # rate for each feature suggested by LASSO
    global X_train
    global y_train
    global X_test
    global y_test

    def acquire_gpu(self):
        # to extend gpu memory while fitting model
        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

    def read_dataset(self):
        # read input file
        file = '/home/mohsen/work-space/Thesis/dataset/CardiacPrediction.xlsx'
        # ipData = pd.read_excel(file, sheet_name='Stroke')
        ipData = pd.read_excel(file, sheet_name='CoroHeartDis')
        CHD_label = ipData.CoronaryHeartDisease
        counter = Counter(CHD_label)
        print("Shape of input dataset = " + str(ipData.shape))
        print("Distribution of input dataset = " + str(counter))
        print(str(len(ipData.index)) + " records")
        return ipData

    def drop_variables(self, ip_data):
        # data drop
        self.opLabel = np.array(ip_data['CoronaryHeartDisease'])
        ip_data.drop(['SEQN', 'CoronaryHeartDisease', 'Annual-Family-Income', 'Height', 'Ratio-Family-Income-Poverty',
                      'X60-sec-pulse',
                      'Health-Insurance', 'Lymphocyte', 'Monocyte', 'Eosinophils', 'Total-Cholesterol', 'Mean-Cell-Vol',
                      'Mean-Cell-Hgb-Conc.', 'Hematocrit', 'Segmented-Neutrophils'], axis=1, inplace=True)
        print("Shape after dropping variables" + str(ip_data.shape))
        # opLabel = np.array(ipData['Stroke'])
        # ipData.drop(['SEQN','Stroke','Annual-Family-Income','Height','Ratio-Family-Income-Poverty','X60-sec-pulse',
        # 'Health-Insurance','Lymphocyte','Monocyte','Eosinophils','Total-Cholesterol','Mean-Cell-Vol',
        # 'Mean-Cell-Hgb-Conc.','Hematocrit','Segmented-Neutrophils'], axis = 1, inplace=True)
        return ip_data

    def convert_to_dummies(self, ip_data):
        # dummy variable for categorical variables
        ip_data = pd.get_dummies(ip_data, columns=["Gender", "Diabetes", "Blood-Rel-Diabetes", "Blood-Rel-Stroke",
                                                   "Vigorous-work", "Moderate-work"])
        self.varb = np.array(ip_data.columns)
        return np.array(ip_data)

    def apply_lasso(self, ip_data):
        feature_votes = np.zeros(ip_data.shape[1])
        iteration = 100
        for num in range(iteration):
            label0_index = np.where(self.opLabel == 0)[0]  # no coronary heart disease
            label1_index = np.where(self.opLabel == 1)[0]  # coronary heart disease
            numTrainData0 = 1300
            numTrainData1 = 1300
            np.random.shuffle(label0_index)
            np.random.shuffle(label1_index)
            label0_index_train = label0_index[0:numTrainData0 - 1]
            label1_index_train = label1_index[0:numTrainData1 - 1]
            # label0_index_test = label0_index[numTrainData0 - 1:]
            # label1_index_test = label1_index[numTrainData1 - 1:]
            # testIndex = np.append(label0_index_test, label1_index_test)
            trainIndex = np.append(label0_index_train, label1_index_train)
            trainData = ip_data[trainIndex]
            trainLabel = self.opLabel[trainIndex]
            # testData = ip_data[testIndex]
            # testLabel = self.opLabel[testIndex]
            scaler = preprocessing.StandardScaler().fit(trainData)
            trainData_scaled = scaler.transform(trainData)
            # testData_scaled = scaler.transform(testData)
            # Elastic net and Lasso from scikit
            # regression = ElasticNet(random_state=0, alpha=1, l1_ratio=0.03, tol=0.000001, max_iter=100000)
            regression = Lasso(random_state=0, alpha=0.006, tol=0.000001, max_iter=100000)
            # regression = LogisticRegression(penalty='l1',random_state=0,C=100,tol=0.000001,max_iter=100,
            # class_weight='balanced')
            regression.fit(trainData_scaled, trainLabel)
            cof = np.abs(regression.coef_)
            colIndex = np.where(cof != 0)[0]
            for col in colIndex:
                feature_votes[col] += 1
        # feature nomination via Lasso (from feature 1 to 30)
        # keep the dummy variables'
        # threshold = iteration // 5  # Pick features occurring more than 5 times
        threshold = 0
        self.featureIndex = np.where(feature_votes[0:30] >= threshold)[0]
        self.featureIndex = np.append(self.featureIndex, np.arange(30, ip_data.shape[1]))
        self.feature_weights = preprocessing.minmax_scale(feature_votes, feature_range=(0, 1))
        print("Selected features shape = " + str(self.featureIndex.shape))
        # ???
        # tInx = np.arange(ip_data.shape[1])
        # rrInx = tInx[~np.isin(tInx, self.featureIndex)]
        # print(self.varb[rrInx])

    def split_dataset(self, ip_data, test_size):
        selected_data = ip_data[:, self.featureIndex]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(selected_data,
                                                                                self.opLabel,
                                                                                test_size=test_size,
                                                                                shuffle=True,
                                                                                stratify=self.opLabel,
                                                                                random_state=5)

    def split_k_fold(self, k, ip_data):
        folds = []
        selected_data = ip_data[:, self.featureIndex]
        kf = KFold(n_splits=k, random_state=None, shuffle=False)
        for train_index, test_index in kf.split(selected_data):
            folds.append([selected_data[train_index],
                          self.opLabel[train_index],
                          selected_data[test_index],
                          self.opLabel[test_index]])

        return folds

    def initialize_model(self, over_sampler):
        selected_weights = self.feature_weights[self.featureIndex]
        # over sample minority class only for the train set.
        self.X_train, self.y_train = self.augment_set(over_sampler, self.X_train, self.y_train)
        # standardize train set and test set values(in rage 0-1).
        self.X_train = preprocessing.minmax_scale(self.X_train, feature_range=(0, 1))
        self.X_test = preprocessing.minmax_scale(self.X_test, feature_range=(0, 1))
        # apply lasso weights on both X-sets
        self.apply_lasso_weights(selected_weights, self.X_train)
        self.apply_lasso_weights(selected_weights, self.X_test)
        # print result
        y_train_counter = Counter(self.y_train)
        y_test_counter = Counter(self.y_test)
        # print shape and distributions
        print("X_train shape = " + str(self.X_train.shape))
        print("y_train shape = " + str(self.y_train.shape))
        print("X_test shape = " + str(self.X_test.shape))
        print("y_test shape = " + str(self.y_test.shape))
        print("y_train distribution = " + str(y_train_counter))
        print("y_test distribution = " + str(y_test_counter))
        # reshape
        self.reshape_sets()

    def augment_set(self, over_sampler, X, y):
        over_sample = OverSample()
        if over_sampler == 'SMOTE':
            X_aug, y_aug = over_sample.smote(X, y)
        elif over_sampler == 'K_MEANS_SMOTE':
            X_aug, y_aug = over_sample.k_means_smote(X, y)
        elif over_sampler == 'SVM_SMOTE':
            X_aug, y_aug = over_sample.svm_smote(X, y)
        elif over_sampler == 'BORDERLINE_SMOTE':
            X_aug, y_aug = over_sample.borderline_smote(X, y)
        elif over_sampler == 'RANDOM':
            X_aug, y_aug = over_sample.random(3, X, y)
        elif over_sampler == 'ADASYN':
            X_aug, y_aug = over_sample.adasyn(X, y)
        else:
            X_aug, y_aug = over_sample.smote(X, y)
        return X_aug, y_aug

    def reshape_sets(self):
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        # one-hot-encoding
        self.y_train = keras.utils.to_categorical(self.y_train, 2)
        self.y_test = keras.utils.to_categorical(self.y_test, 2)

    def shuffle_dataset(self, X, y):
        from sklearn.utils import shuffle
        data = self.merge_label(X, y)
        data = shuffle(data)
        return self.split_label(data)

    def merge_label(self, X, y):
        return np.concatenate((X, y.T[:, None]), axis=1)

    def split_label(self, data):
        x_set = data[:, :-1]
        y = data[:, -1]
        return x_set, y

    def apply_lasso_weights(self, weights, dataset):
        for n, w in enumerate(weights):
            dataset[:, n] *= w
            n += 1

    def visualize_sets(self):
        X_embedded = TSNE(n_components=3, n_iter=300, verbose=1).fit_transform(self.X_train)
        cin = sns.color_palette("Set1")[1]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('w')
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],
                   c=[sns.color_palette("Set1")[x] for x in self.y_train],
                   edgecolors='k', label='no-CHD')
        ax.scatter(0, 0, 0, c=cin, edgecolors='k', label='CHD')
        ax.azim = 20
        ax.elev = 20
        ax.set_xlabel("t-SNE Dim 1", size="x-large")
        ax.set_ylabel("t-SNE Dim 2", size="x-large")
        ax.set_zlabel("t-SNE Dim 3", size="x-large")
        plt.title("Random subsampling 3:1", size="xx-large")
        ax.legend(loc='upper left')
        fig.set_size_inches(10, 10)
        fig.savefig('tSNE_RUS.png', dpi=100)

    def create_model(self, lr):
        inputs = keras.layers.Input(shape=(self.X_train.shape[1], 1))
        # Dense layer
        RS0 = keras.layers.Reshape((self.X_train.shape[1],))(inputs)
        FC0 = keras.layers.Dense(128, bias_initializer=keras.initializers.VarianceScaling())(RS0)
        BN0 = keras.layers.BatchNormalization(axis=-1)(FC0)
        AC0 = keras.layers.Activation('relu')(BN0)
        DP0 = keras.layers.Dropout(0.2)(AC0)
        # conv layer 1
        RS1 = keras.layers.Reshape((128, 1))(DP0)
        FC1 = keras.layers.Conv1D(2, 3, strides=1)(RS1)
        BN1 = keras.layers.BatchNormalization(axis=-1)(FC1)
        AC1 = keras.layers.Activation('relu')(BN1)
        Pool1 = keras.layers.AveragePooling1D(pool_size=2)(AC1)
        # conv layer 2
        FC2 = keras.layers.Conv1D(4, 5, strides=1)(Pool1)
        BN2 = keras.layers.BatchNormalization(axis=-1)(FC2)
        AC2 = keras.layers.Activation('relu')(BN2)
        Pool2 = keras.layers.AveragePooling1D(pool_size=2)(AC2)
        # conv layer 3
        FC2_3 = keras.layers.Conv1D(4, 5, strides=1)(Pool2)
        BN2_3 = keras.layers.BatchNormalization(axis=-1)(FC2_3)
        AC2_3 = keras.layers.Activation('relu')(BN2_3)
        Pool2_3 = keras.layers.AveragePooling1D(pool_size=2)(AC2_3)
        # conv layer 4
        FC2_4 = keras.layers.Conv1D(4, 5, strides=1)(Pool2_3)
        BN2_4 = keras.layers.BatchNormalization(axis=-1)(FC2_4)
        AC2_4 = keras.layers.Activation('relu')(BN2_4)
        Pool2_4 = keras.layers.AveragePooling1D(pool_size=2)(AC2_4)
        # flatten
        FL1 = keras.layers.Flatten()(Pool2_4)
        # output Dense
        FC3 = keras.layers.Dense(512, bias_initializer=keras.initializers.VarianceScaling())(FL1)
        BN3 = keras.layers.BatchNormalization(axis=-1)(FC3)
        AC3 = keras.layers.Activation('relu')(BN3)
        DP3 = keras.layers.Dropout(0.2)(AC3)
        # output Dense
        FC4 = keras.layers.Dense(2)(DP3)
        outputs = keras.layers.Activation('sigmoid')(FC4)
        # compile model
        CNN1D4 = keras.Model(inputs=inputs, outputs=outputs)
        CNN1D4.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=[tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall(),
                     tf.keras.metrics.AUC()]
        )
        # CNN1D4.summary()
        return CNN1D4

    def fit_model(self, model, epochs, monitor, mode):
        callback = tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=0,
            patience=50,
            verbose=0,
            mode=mode,
            baseline=None,
            restore_best_weights=False,
        )
        return model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            callbacks=[callback],
            validation_data=(self.X_test, self.y_test)
        )

    def evaluate_model(self, model):
        val_loss, val_precision, val_recall, val_auc = model.evaluate(x=self.X_test, y=self.y_test)
        y_predict = model.predict(self.X_test)
        conf_mat = metrics.confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(y_predict, axis=1))
        keras.backend.clear_session()
        return val_loss, val_precision, val_recall, val_auc, conf_mat

    def show_result(self, history):
        self.show_history(history, 'precision')
        self.show_history(history, 'auc')
        self.show_history(history, 'recall')
        self.show_history(history, 'loss')
        self.show_history_diff(history, 'precision')
        self.show_history_diff(history, 'auc')
        self.show_history_diff(history, 'recall')
        self.show_history_diff(history, 'loss')

    def show_history(self, train_res, metric):
        plt.plot(train_res.history[metric])
        plt.plot(train_res.history['val_' + metric])
        plt.title('model ' + metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
        plt.show()

    def show_history_diff(self, train_res, metric):
        diff = list(np.array(train_res.history[metric]) - np.array(train_res.history['val_' + metric]))
        plt.plot(diff)
        plt.title('model ' + metric + ' diff')
        plt.ylabel('difference')
        plt.xlabel('epoch')
        plt.show()

    def start_learning(self, lr, epochs, callback_monitor):
        cnn_model = self.create_model(lr=lr)
        self.fit_model(model=cnn_model,
                       epochs=epochs,
                       monitor=callback_monitor,
                       mode="max")
        return chd_prediction.evaluate_model(model=cnn_model)

    def report_result(self, eval_results):
        losses = [item[0] for item in eval_results]
        precisions = [item[1] for item in eval_results]
        recalls = [item[2] for item in eval_results]
        AUCs = [item[3] for item in eval_results]
        mean_loss = statistics.mean(losses)
        mean_precision = statistics.mean(precisions)
        mean_recall = statistics.mean(recalls)
        mean_auc = statistics.mean(AUCs)
        var_loss = np.var(losses)
        var_precision = np.var(precisions)
        var_recall = np.var(recalls)
        var_auc = np.var(AUCs)
        print("**************************************** Result Report ****************************************")
        print("Average Loss  = ", mean_loss)
        print("Average Precision = ", mean_precision)
        print("Average Recall = ", mean_recall)
        print("Average AUC = ", mean_auc)
        print("Variance of Loss = ", var_loss)
        print("Variance of Precision = ", var_precision)
        print("Variance of Recall = ", var_recall)
        print("Variance of AUC = ", var_auc)


if __name__ == '__main__':
    chd_prediction = CHDPredictionCV()
    chd_prediction.acquire_gpu()
    input_data = chd_prediction.read_dataset()
    input_data = chd_prediction.drop_variables(input_data)
    input_data = chd_prediction.convert_to_dummies(input_data)
    chd_prediction.apply_lasso(input_data)
    k_folds = chd_prediction.split_k_fold(3, input_data)
    folds_eval_result = []
    for i, fold in enumerate(k_folds):
        chd_prediction.X_train = fold[0]
        chd_prediction.y_train = fold[1]
        chd_prediction.X_test = fold[2]
        chd_prediction.y_test = fold[3]
        print("")
        print("**************************************** Fold Number", i + 1,
              "Model Initialization ****************************************")
        chd_prediction.initialize_model('SVM_SMOTE')
        print("")
        print("**************************************** Fold Number", i + 1,
              "Learning Procedure ******************************************")
        result = chd_prediction.start_learning(lr=0.01, epochs=2, callback_monitor='precision')
        folds_eval_result.append(result)
    chd_prediction.report_result(folds_eval_result)
