#!/usr/bin/env python
# coding: utf-8

# Youtube spam detection (Classiffication)

# load the libraries
# You may need install some packages from nltk after import nltk ()
import numpy as np
import matplotlib

# for showing images comment out next command
matplotlib.use("Agg")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import joblib
import nltk

# from supervised import Classifiers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
import sklearn
from sklearn.model_selection import KFold
import scipy.stats as stats
import os

# from nltk.corpus import stopwords
import re
from collections import Counter
import itertools
import xgboost
import catboost
import lightgbm
import keras
from keras.utils import np_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import LSTM, Input, Dense, GRU
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras import regularizers
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import keras_metrics
import sklearn
import pickle
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import RepeatVector, TimeDistributed, Flatten
from keras import initializers, optimizers
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from functools import reduce
import operator
from keras.callbacks import Callback
from sklearn import metrics
import sys
import collections
import pickle
import collections
from matplotlib import pyplot as plt
import logging
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import random
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
from sklearn.covariance import EmpiricalCovariance
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import Birch
from sklearn.mixture import BayesianGaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import time
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
)
import warnings
import xgboost
import catboost
import lightgbm

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)


def main(Psy_df, KatyPerry_df, LMFAO_df, Eminem_df, Shakira_df, dl_model):

    ##################################### LOADING AND VISUALIZATION ###################################

    # concatenate all df to create all_df dataframe
    all_df = pd.concat([Psy_df, KatyPerry_df, LMFAO_df, Eminem_df, Shakira_df])

    # Create a dictionary with all the dataframes
    dfs = {
        "Psy_df": Psy_df,
        "KP_df": KatyPerry_df,
        "LMFAO_df": LMFAO_df,
        "Eminem_df": Eminem_df,
        "shakira_df": Shakira_df,
        "all_df": all_df,
    }

    # show the first 5 rows of the all_df dataframe
    print(all_df)

    # Describe all_df dataframe
    print(all_df.info())

    # length of characters in all_df dataframe
    all_df["CONTENT_LEN"] = all_df["CONTENT"].str.len()
    print(all_df["CONTENT_LEN"].describe())
    print(all_df.groupby("CLASS").describe())

    # Plot boxplot for all_df dataframe show count vs classes
    plt.figure()
    ax = sns.boxplot(x="CLASS", y="CONTENT_LEN", data=all_df)
    ax.set(xlabel="all_df", ylabel="Count")
    plt.show()

    # plot distplot show length vs classes for all dataframes
    # data is balanced between two classes
    for name, data in dfs.items():
        plt.figure()
        ax = sns.distplot(data["CLASS"], bins=4, kde=False)
        ax.set(xlabel=name, ylabel="Length")
        plt.show()

    # Plot relationship between author and classes for all_df dataframe
    pd.crosstab(all_df["AUTHOR"], all_df["CLASS"]).plot(kind="bar")
    plt.title("Relationship between author and classes")
    plt.xlabel("Author")
    plt.ylabel("Count classes")
    plt.show()

    ####################################### ALL MACHINE LEARNING MODELS ########################################

    CLASSIFIERS = [
        est for est in all_estimators() if issubclass(est[1], ClassifierMixin)
    ]

    removed_classifiers = [
        ("ClassifierChain", sklearn.multioutput.ClassifierChain),
        ("ComplementNB", sklearn.naive_bayes.ComplementNB),
        ("MultiOutputClassifier", sklearn.multioutput.MultiOutputClassifier),
        ("OneVsOneClassifier", sklearn.multiclass.OneVsOneClassifier),
        ("OneVsRestClassifier", sklearn.multiclass.OneVsRestClassifier),
        ("OutputCodeClassifier", sklearn.multiclass.OutputCodeClassifier),
        ("StackingClassifier", sklearn.ensemble.StackingClassifier),
        ("VotingClassifier", sklearn.ensemble.VotingClassifier),
    ]

    for i in removed_classifiers:
        CLASSIFIERS.pop(CLASSIFIERS.index(i))

    CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
    CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))
    # CLASSIFIERS.append(("CatBoostClassifier", catboost.CatBoostClassifier))

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer_low = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    categorical_transformer_high = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            ("encoding", OrdinalEncoder()),
        ]
    )

    def get_rmvar(train_x, test_x, threshold=0.01):
        """Remove features with low variance.

        Args:
        train_x (numpy ndarray): x_train data
        test_x (numpy ndarray): X_test data
        threshold (float, optional): threshold for variance. Defaults to 0.01.

        Returns:
            numpy ndarray: X_train and X_test data with removed features.
        """
        print("train_x before remove var", train_x.shape)
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(train_x)
        train_var = selector.transform(train_x)
        test_var = selector.transform(test_x)
        print("train_x after remove var", train_x.shape)

        return train_var, test_var

    # Helper function
    def get_card_split(df, cols, n=11):
        """
        Splits categorical columns into 2 lists based on cardinality (i.e # of unique values)

        Args:
        df (Pandas DataFrame): DataFrame from which the cardinality of the columns is calculated.
        cols (list): Categorical columns to list
        n (int, optional (default=11)): The value of 'n' will be used to split columns.

        Returns:
        card_low (list): Columns with cardinality < n
        card_high (list): Columns with cardinality >= n
        """
        cond = df[cols].nunique() > n
        card_high = cols[cond]
        card_low = cols[~cond]
        return card_low, card_high

    class Classifiers:
        """
        This module helps in fitting to all the classification algorithms that are available in Scikit-learn

        Args:
        verbose (int, optional (default=0)): For the liblinear and lbfgs solvers set verbose to any positive
            number for verbosity.
        ignore_warnings (bool, optional (default=True)): When set to True, the warning related to algorigms
            that are not able to run are ignored.
        custom_metric (function, optional (default=None)): When function is provided, models are evaluated
            based on the custom evaluation metric provided.
        prediction (bool, optional (default=False)): When set to True, the predictions of all the models
            models are returned as dataframe.
        classifiers (list, optional (default="all")): When function is provided, trains the chosen classifiers.
        """

        def __init__(
            self,
            verbose=0,
            ignore_warnings=True,
            custom_metric=None,
            predictions=False,
            random_state=42,
            classifiers="all",
        ):
            self.verbose = verbose
            self.ignore_warnings = ignore_warnings
            self.custom_metric = custom_metric
            self.predictions = predictions
            self.models = {}
            self.random_state = random_state
            self.classifiers = classifiers

        def fit(self, X_train, X_test, y_train, y_test):
            """Fit Classification algorithms to X_train and y_train, predict and score on X_test, y_test.

            Args:
            X_train (array): Training vectors, where rows is the number of samples
                and columns is the number of features.
            X_test (array): Testing vectors, where rows is the number of samples
                and columns is the number of features.
            y_train (array): Training vectors, where rows is the number of samples
                and columns is the number of features.
            y_test (array): Testing vectors, where rows is the number of samples
                and columns is the number of features.

            Returns:
            scores (Pandas DataFrame): Returns metrics of all the models in a Pandas DataFrame.
            predictions (Pandas DataFrame): Returns predictions of all the models in a Pandas DataFrame.
            """
            Accuracy = []
            B_Accuracy = []
            ROC_AUC = []
            F1 = []
            names = []
            TIME = []
            predictions = {}

            if self.custom_metric is not None:
                CUSTOM_METRIC = []

            if isinstance(X_train, np.ndarray):
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)

            numeric_features = X_train.select_dtypes(
                include=[np.number]
            ).columns
            categorical_features = X_train.select_dtypes(
                include=["object"]
            ).columns

            categorical_low, categorical_high = get_card_split(
                X_train, categorical_features
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric", numeric_transformer, numeric_features),
                    (
                        "categorical_low",
                        categorical_transformer_low,
                        categorical_low,
                    ),
                    (
                        "categorical_high",
                        categorical_transformer_high,
                        categorical_high,
                    ),
                ]
            )

            if self.classifiers == "all":
                self.classifiers = CLASSIFIERS
            else:
                try:
                    temp_list = []
                    for classifier in self.classifiers:
                        full_name = (classifier.__class__.__name__, classifier)
                        temp_list.append(full_name)
                    self.classifiers = temp_list
                except Exception as exception:
                    print(exception)
                    print("Invalid Classifier(s)")

            for name, model in tqdm(self.classifiers):
                start = time.time()
                try:
                    if "random_state" in model().get_params().keys():
                        pipe = Pipeline(
                            steps=[
                                ("preprocessor", preprocessor),
                                (
                                    "classifier",
                                    model(random_state=self.random_state),
                                ),
                            ]
                        )
                    else:
                        pipe = Pipeline(
                            steps=[
                                ("preprocessor", preprocessor),
                                ("classifier", model()),
                            ]
                        )

                    pipe.fit(X_train, y_train)
                    self.models[name] = pipe
                    y_pred = pipe.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred, normalize=True)
                    b_accuracy = balanced_accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted")
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred)
                    except Exception as exception:
                        roc_auc = None
                        if self.ignore_warnings is False:
                            print("ROC AUC couldn't be calculated for " + name)
                            print(exception)
                    names.append(name)
                    Accuracy.append(accuracy)
                    B_Accuracy.append(b_accuracy)
                    ROC_AUC.append(roc_auc)
                    F1.append(f1)
                    TIME.append(time.time() - start)
                    if self.custom_metric is not None:
                        custom_metric = self.custom_metric(y_test, y_pred)
                        CUSTOM_METRIC.append(custom_metric)
                    if self.verbose > 0:
                        if self.custom_metric is not None:
                            print(
                                {
                                    "Model": name,
                                    "Accuracy": accuracy,
                                    "Balanced Accuracy": b_accuracy,
                                    "ROC AUC": roc_auc,
                                    "F1 Score": f1,
                                    self.custom_metric.__name__: custom_metric,
                                    "Time taken": time.time() - start,
                                }
                            )
                        else:
                            print(
                                {
                                    "Model": name,
                                    "Accuracy": accuracy,
                                    "Balanced Accuracy": b_accuracy,
                                    "ROC AUC": roc_auc,
                                    "F1 Score": f1,
                                    "Time taken": time.time() - start,
                                }
                            )
                    if self.predictions:
                        predictions[name] = y_pred
                except Exception as exception:
                    if self.ignore_warnings is False:
                        print(name + " model failed to execute")
                        print(exception)
            if self.custom_metric is None:
                scores = pd.DataFrame(
                    {
                        "Model": names,
                        "Accuracy": Accuracy,
                        "Balanced Accuracy": B_Accuracy,
                        "ROC AUC": ROC_AUC,
                        "F1 Score": F1,
                        "Time Taken": TIME,
                    }
                )
            else:
                scores = pd.DataFrame(
                    {
                        "Model": names,
                        "Accuracy": Accuracy,
                        "Balanced Accuracy": B_Accuracy,
                        "ROC AUC": ROC_AUC,
                        "F1 Score": F1,
                        self.custom_metric.__name__: CUSTOM_METRIC,
                        "Time Taken": TIME,
                    }
                )
            scores = scores.sort_values(
                by="Balanced Accuracy", ascending=False
            ).set_index("Model")

            if self.predictions:
                predictions_df = pd.DataFrame.from_dict(predictions)
            return (
                scores,
                predictions_df if self.predictions is True else scores,
            )

    # Helper class for performing classification
    Classification = Classifiers

    ################################# MACHINE LEARNING FOR BEST MODEL ####################################

    best_model = []

    def models_selection(X_train, X_test, y_train, y_test, name, base, clf):
        """train and test model with different ML models without cross validation.

        Args:
            X_train (numpy ndarray): X train data
            X_test (numpy ndarray): X test data
            y_train (numpy ndarray): y train data or classes
            y_test (numpy ndarray): y test data or classes
            name (str): name of the model
            base (bool): if True, use base model
            clf (class): if passed then a base model is trained and test to see the results

        Returns:
            list of pandas dataframe: best models with sorted accuracies from all models for all dataframes
        """

        if base:
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            pred = clf.predict(X_test)
            roc = roc_auc_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="weighted")
            print(f"Accuracy: {score:.3f}%")
            print(f"ROC score: {roc:.3f}%")
            print(f"F1 score: {f1:.3f}%")
            print("Confusion Matrix: \n", confusion_matrix(pred, y_test))
            print("Confusion Matrix: \n", classification_report(pred, y_test))
        scores, pred_df = Classifiers().fit(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
        )
        scores.to_csv(f"./save_csv/{name}_scores_before_cv.csv")
        print(scores)
        print(
            f"Best model for {name} is {scores.index[0]} with {scores['Accuracy'][0]:.3f}% accuracy"
        )
        best_df = pd.DataFrame(
            {
                "Dataset": [name],
                "Best Model": [scores.index[0]],
                "Accuracy": [scores["Accuracy"][0]],
                "ROC": [scores["ROC AUC"][0]],
                "F1 Score": [scores["F1 Score"][0]],
            }
        )
        best_model.append(best_df)

        return best_model

    for name, dataset in dfs.items():
        # bow and tfidf for dataset
        X_train, X_test, y_train, y_test = train_test_split(
            dataset["CONTENT"].values,
            dataset["CLASS"].values,
            test_size=0.3,
            shuffle=True,
            random_state=42,
        )
        # cv = CountVectorizer(analyzer= "word", stop_words="english").fit(X_train)
        # bow = cv.transform(X_train)
        # X_train = cv.transform(X_train)
        # X_test = cv.transform(X_test)
        # tfidf = TfidfTransformer().fit(bow)
        # tfidf.transform(bow)
        tf_idf_vec = TfidfVectorizer()
        X_train = tf_idf_vec.fit_transform(X_train)
        X_test = tf_idf_vec.transform(X_test)
        print("Data Set:", name)
        best_model = models_selection(
            X_train.toarray(),
            X_test.toarray(),
            np.array(y_train),
            np.array(y_test),
            name,
            base=False,
            clf=ExtraTreesClassifier(),
        )
        print(f"Best model selection (ML) with all ML models for {name}")
        print("********" * 20)

    # Concatenate and save best model df and accuraies without cross validation
    best_model_without_cv = pd.concat(best_model)
    best_model_without_cv.to_csv("./save_csv/best_model_before_cv.csv")
    best_model_without_cv

    models_save = []

    acc_all = []
    roc_all = []
    f1_all = []

    def data_generator(X_train, X_test, dataset_name):
        """Generate data for cross validation.

        Args:
            X_train (numpy ndarray): X train data
            X_test (numpy ndarray): X test data
            dataset_name (str): name of the dataset dataframe

        Returns:
            numpy ndarray: X train and X test data
        """
        for name, dataset in dfs.items():
            if name == dataset_name:
                # bow and tfidf for dataset
                # cv = CountVectorizer(analyzer= "word", stop_words="english").fit(X_train)
                # bow = cv.transform(X_train)
                # X_train = cv.transform(X_train)
                # X_test = cv.transform(X_test)
                # tfidf = TfidfTransformer().fit(bow)
                # tfidf.transform(bow)
                tf_idf_vec = TfidfVectorizer()
                X_train = tf_idf_vec.fit_transform(X_train)
                X_test = tf_idf_vec.transform(X_test)
                return X_train.toarray(), X_test.toarray()

    def dataset_selector(dataset_name):
        for name, dataset in dfs.items():
            if name == dataset_name:
                return dataset

    def models_cv(X, y, name, clf, k_fold):
        """train and test model with different ML models with cross validation
        (based on best models from models_selection).

        Args:
            X (numpy ndarray): X data
            y (numpy ndarray): y data or classes
            name (str): name of the dataset dataframe
            clf (class): classifier model
            k_fold (int): number of folds for cross validation

        Returns:
            list of pandas dataframe: best models with sorted accuracies from all models for all dataframes
        """
        print("Data set:", name)
        print("Classifier:", clf)
        i = 1
        # Cross validation
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=0)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train, X_test = data_generator(X_train, X_test, dataset)
            y_train, y_test = np.array(y_train), np.array(y_test)
            print(
                f"Train and test the Selected Best Model (ML) with Cross Validation for {name} with {clf}"
            )
            print("Fold number: ", i)
            print(X.shape, y.shape)

            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            pred = clf.predict(X_test)
            roc = roc_auc_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="weighted")
            print(f"Accuracy: {score:.3f}%")
            print(f"ROC score: {roc:.3f}%")
            print("Confusion Matrix: \n", confusion_matrix(pred, y_test))
            print("Confusion Matrix: \n", classification_report(pred, y_test))
            acc_all.append(score)
            roc_all.append(roc)
            f1_all.append(f1)
            i += 1

        best_df = pd.DataFrame(
            {
                "Dataset": [name],
                "Best Model": [clf],
                "Accuracy": [np.mean(acc_all)],
                "ROC AUC": [np.mean(roc_all)],
                "F1 Score": [np.mean(f1_all)],
            }
        )
        models_save.append(best_df)

        return models_save

    for dataset, model_best in zip(
        best_model_without_cv["Dataset"], best_model_without_cv["Best Model"]
    ):
        for name, model in CLASSIFIERS:
            if model_best == name:
                dataset_ = dataset_selector(dataset)
                models_cv(
                    X=dataset_["CONTENT"].values,
                    y=dataset_["CLASS"].values,
                    name=dataset,
                    clf=model(),
                    k_fold=10,
                )
                print("********" * 20)

    # Concatenate best model df and accuraies with cross validation
    best_df = pd.concat(models_save)
    # best_df

    # models_save = []
    # score = models(X, y, all_df, RandomForestClassifier(), k_fold=10)
    # print(score)

    #################################### DEEP LEARNING PREPROCESSING ###################################

    class TerminateOnBaseline(Callback):
        """Callback that terminates training when either acc or val_acc reaches a specified baseline

        Args:
            Callback (class): Abstract base class used to build new callbacks.
        """

        def __init__(self, monitor="val_loss", baseline=0.01):
            super(TerminateOnBaseline, self).__init__()
            self.monitor = monitor
            self.baseline = baseline

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            acc = logs.get(self.monitor)
            if acc is not None:
                if acc <= self.baseline:
                    print(
                        "Epoch %d: Reached baseline, terminating training"
                        % (epoch)
                    )
                    self.model.stop_training = True

    # Preprocessing for Deep Learning models
    # random.seed(42)
    # Load dataset
    dataset = all_df[["CONTENT", "CLASS"]]

    # Helper Functions for Text Cleaning
    def string_cleaner(string):
        """String cleaning for dataset

        Args:
            string (str): string from dataset

        Returns:
            str: cleaned strings
        """

        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " 's", string)
        string = re.sub(r"\'ve", " 've", string)
        string = re.sub(r"n\'t", " n't", string)
        string = re.sub(r"\'re", " 're", string)
        string = re.sub(r"\'d", " 'd", string)
        string = re.sub(r"\'ll", " 'll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", "", string)
        string = re.sub(r"\)", "", string)
        string = re.sub(r"\?", "", string)
        string = re.sub(r"/", "", string)
        string = re.sub(r"\s{2,}", " ", string)
        # print (string)
        return string.strip().lower()

    dataset["CONTENT"] = dataset["CONTENT"].apply(string_cleaner)

    # count word frequency for dataset
    word_frequency = {}

    def tokenizer_df(col):
        """create word frequency

        Args:
            col (pandas dataframe): pandas dataframe dataset

        Returns:
            list: tokenized words
        """
        datasetx = col["CONTENT"]
        # tokenize each word in dataset
        tokenize = nltk.wordpunct_tokenize(datasetx)
        tokenize_list = []
        for tokenz in tokenize:
            tokenize_list.append(tokenz.lower())
            if tokenz.lower() in word_frequency:
                cnt = word_frequency[tokenz.lower()]
                cnt += cnt
                word_frequency[tokenz.lower()] = cnt
            else:
                word_frequency[tokenz.lower()] = 1
        return ",".join(tokenize_list)

    def remove_numerics(col):
        """remove numbers from sentences

        Args:
            col (pandas dataframe): pandas dataframe

        Returns:
            list: list of cleaned sentences
        """
        datasetx = col["CONTENT"]
        if type(datasetx) not in [int, float]:
            each_line = re.sub(r"[^A-Za-z\s]", " ", datasetx.strip())
            tokenize = each_line.split()
        else:
            tokenize = []
        return " ".join(tokenize)

    # get sentences, remove numbers, get word frequency, tokenize it and merge similar words
    similarity_word_reduce = WordNetLemmatizer()

    def get_ds(dataset):
        """receive dataset and perform tokenization and lemmatization

        Args:
            dataset (pandas dataframe): pandas dataframe dataset

        Returns:
            pandas dataframe: pandas datarame with tokenized and lemmatized sentences
        """
        datasetx = dataset
        datasetx["CONTENT"] = datasetx.apply(remove_numerics, axis=1)
        datasetx["tokenize"] = datasetx.apply(tokenizer_df, axis=1)
        # print ("dataset head in received", datasetx.head())
        n = []
        for i in datasetx["tokenize"]:
            list = []
            for j in i.split(","):
                list.append(similarity_word_reduce.lemmatize(j))
            n.append(" ".join(list))
        datasetx["Summary_lemmatize"] = n
        return datasetx["Summary_lemmatize"]

    dataset["CONTENT"] = get_ds(dataset[["CONTENT"]])

    print(dataset.head())

    # Cut sentences in dataset
    list_all = []
    sentence_len = 30
    for i in dataset.values:
        if len(i[0]) <= sentence_len and len(i[0]) >= 1:
            list_all.append([i[0], i[1]])

        elif len(i[0]) > sentence_len:
            tmp = " ".join(i[0].split()[0:sentence_len])
            list_all.append([tmp, i[1]])

    # print("length of list and dataset in received func", len(list_all),len(dataset))

    # make our dataset again
    dataset = pd.DataFrame(list_all, columns=["CONTENT", "CLASS"])
    # print("datasetset df, length  : \n", dataset.head())
    # print(dataset)
    print(
        "number of positive sentences: ",
        len(dataset[dataset["CLASS"] == 1]),
        " ,number of negative sentences: ",
        len(dataset[dataset["CLASS"] == 0]),
    )

    # load x and labels
    def load_x_y():
        # Load dataset from files
        # Split by words
        x_all = dataset["CONTENT"].to_numpy()
        x_all = [string_cleaner(sent) for sent in x_all]
        # print(
        #     "x_test  ", x_all[0]
        # )
        x_all = [s.split(" ") for s in x_all]

        y_all = dataset["CLASS"].to_numpy()
        return [x_all, y_all]

    # pad all sentences in case all sentences have the same length
    sentences_length = sentence_len

    def padding(sequences, pad="<PWD/>"):
        """pad sentences to have same length

        Args:
            sequences (numpy ndaaray): sentences from dataset
            pad (str, optional): pad token if sentence is short. Defaults to "<PWD/>".

        Returns:
            numpy ndarray: padded sentences
        """
        padded_sentences = []
        for i in range(len(sequences)):
            sentence = sequences[i]
            pad_no = sentences_length - len(sentence)
            if pad_no <= 0:
                tmp = sentence[0 : sentences_length - 1]
                tmp.append(pad)
                new_sequence = " ".join(tmp).split()
            else:
                new_sequence = sentence + [pad] * pad_no

            padded_sentences.append(new_sequence)
        # print ("padded_sentences and length of it in padding func",
        # padded_sentences[0],len(padded_sentences[0]))
        return padded_sentences

    def vocab_builder(sequences):
        """build vocabulary, word2index and index2word

        Args:
            sequences (numpy ndarray): sentences from dataset

        Returns:
            list: vocab and vocab invarient
        """
        word_cnt = Counter(itertools.chain(*sequences))
        vocab_invariant = [x[0] for x in word_cnt.most_common()]
        vocabs = {x: i for i, x in enumerate(vocab_invariant)}
        # print("vocabs", vocabs)
        # print("vocab_invariant", vocab_invariant)
        return [vocabs, vocab_invariant]

    # map sentences
    def mapper(sequences, labels, vocabs):
        x = np.array(
            [[vocabs[str(word)] for word in sentence] for sentence in sequences]
        )
        y = np.array(labels)
        return [x, y]

    def dataset_loader():
        """load x, labels, vocab and vocab invariants

        Returns:
            list: x and y data, vocab and vocab invariant
        """
        sequences, labels = load_x_y()

        print("x sequence 0 : ", sequences[0])
        sentences_padded = padding(sequences)
        # print("x sequences 0 to 1 after padding (sequence length): ", sequences_padded[0:2])
        vocabs, vocab_invariant = vocab_builder(sentences_padded)
        x, y = mapper(sentences_padded, labels, vocabs)
        return [x, y, vocabs, vocab_invariant]

    # Load data after preprocessing
    print("Loading data...")
    x, y, vocabs, vocab_invariant = dataset_loader()
    # x = keras.preprocessing.sequence.pad_sequences(
    #     x, maxlen=sentence_len, truncating="post", padding="post"
    # )

    # min_max_scaler = preprocessing.MinMaxScaler()
    # x = min_max_scaler.fit_transform(x)

    # print("vocab_invariant: ", vocab_invariant)
    print("vocabs size: {:d}".format(len(vocabs)))
    # print("vocabs : ", vocabs)
    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape)
    print("x.dtype: ", x.dtype)
    print(x[100])

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
    # X_train, X_test = get_rmvar(X_train, X_test, threshold=0.1)

    #################################### DEEP LEARNING TRAINING ###################################

    # LSTM with cross valdation

    acc_all = []
    roc_all = []
    f1_all = []

    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    i = 1
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_shuffled = to_categorical(y_train)
        y_shuffled_test = to_categorical(y_test)

        print(f"Fold number: {i}")
        embedding_length = 4
        model = Sequential()

        model.add(
            Embedding(5000, embedding_length, input_length=X_train.shape[1])
        )
        if dl_model == "LSTMmodel":
            model.add(Dropout(0.5))
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.5))
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.5))
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.5))
            model.add(LSTM(64))
            model.add(Dense(2, activation="softmax"))
        else:
            model.add(Dropout(0.5))
            model.add(GRU(64, return_sequences=True))
            model.add(Dropout(0.5))
            model.add(GRU(64, return_sequences=True))
            model.add(Dropout(0.5))
            model.add(GRU(64, return_sequences=True))
            model.add(Dropout(0.5))
            model.add(GRU(64))
            model.add(Dense(2, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics="accuracy",
        )
        print(model.summary())

        save_best = keras.callbacks.ModelCheckpoint(
            "./saved_model/model.h5",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
        )
        # Train phase
        hist = model.fit(
            X_train,
            y_shuffled,
            epochs=100,
            validation_split=0.1,
            batch_size=16,
            shuffle=True,
            verbose=2,
            callbacks=[
                TerminateOnBaseline(monitor="val_loss", baseline=0.01),
                save_best,
            ],
        )

        # Testing phase
        accuray = model.evaluate(
            X_test, y_shuffled_test, verbose=2, batch_size=32
        )
        # print("Test score: %.2f%%" % (scores))
        print("Test Accuracy: ", (accuray[1]))

        # model = load_model('./saved_model/model.h5')
        y_pred = model.predict(X_test, batch_size=32, verbose=1)
        y_pred_bool = np.argmax(y_pred, axis=1)

        # print(y_pred_bool)
        print(collections.Counter(y_pred_bool))
        # print(type(y_shuffled_test))
        # y_pred_bool=list(y_pred_bool)
        # y_shuffled_test=list(y_shuffled_test)
        print(classification_report(y_test, y_pred_bool))
        cm = confusion_matrix(y_test, y_pred_bool)
        print(cm)
        roc = roc_auc_score(y_test, y_pred_bool)
        f1 = f1_score(y_test, y_pred_bool, average="weighted")
        acc_all.append(accuray[1])
        roc_all.append(roc)
        f1_all.append(f1)

        i += 1
        print("**********" * 20)

    print("LSTM Average Test Accuracy from cv: ", np.mean(acc_all))

    # Load all_df results before cv
    ordered_models_without_cross_validation = pd.read_csv(
        "./save_csv/all_df_scores_before_cv.csv"
    )
    print("\nmodels without cross validation (except LSTM) for all_df: ")
    print(ordered_models_without_cross_validation)

    # Append lstm to best model with cv and save it as csv
    best_model_cv = best_df.append(
        {
            "Dataset": "all_df",
            "Best Model": dl_model,
            "Accuracy": np.mean(acc_all),
            "ROC AUC": np.mean(roc_all),
            "F1 Score": np.mean(f1_all),
        },
        ignore_index=True,
    )
    best_model_cv.to_csv("./save_csv/best_model_after_cv.csv")
    print("\nBest models with cross validation: ")
    print(best_model_cv)

    # Show the best model and accuracy based on all models traned and tested
    all_df_results = best_model_cv[best_model_cv["Dataset"] == "all_df"]
    best_model_in_all = all_df_results[
        all_df_results["Accuracy"] == all_df_results["Accuracy"].max()
    ]
    print(
        "\nBest model with cross validation based on highest accuracy for all_df: "
    )
    print(best_model_in_all)

    # Time elapesed for whole processes
    print("\nTotal Time (in seconds) =", (time.time() - start))


if __name__ == "__main__":

    start = time.time()
    print("Start")

    # Load data
    Psy_df = pd.read_csv("./data/Youtube01-Psy.csv")
    KatyPerry_df = pd.read_csv("./data/Youtube02-KatyPerry.csv")
    LMFAO_df = pd.read_csv("./data/Youtube03-LMFAO.csv")
    Eminem_df = pd.read_csv("./data/Youtube04-Eminem.csv")
    Shakira_df = pd.read_csv("./data/Youtube05-Shakira.csv")

    # dl_model (str) can be "LSTMmodel" or "GRUmodel"
    # All ML models are used as default
    main(
        Psy_df,
        KatyPerry_df,
        LMFAO_df,
        Eminem_df,
        Shakira_df,
        dl_model="GRUmodel",
    )

    # TODO:
    # hyperparameter tuning parameters
    # clean code
    # unit testing
    # anomalies detection
