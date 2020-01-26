import re
import warnings

# Visualization
import matplotlib as mpl
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
# NumPy
import numpy as np
# Load in our libraries
# Dataframe operations
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
# Data visualization
import seaborn as sns
import sklearn
import xgboost as xgb
from pandas.plotting import scatter_matrix
import random
# Common Model Algorithms
from sklearn import metrics  # accuracy measure
from sklearn import svm  # support vector Machine
from sklearn import (discriminant_analysis, ensemble, feature_selection,
                     gaussian_process, linear_model, model_selection,
                     naive_bayes, neighbors, tree)
from sklearn.model_selection import KFold
# Going to use these 5 base models for the stacking
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, VotingClassifier)
# Models
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix  # for confusion matrix
from sklearn.metrics import accuracy_score, mean_squared_error
# GridSearchCV
# Cross-validation
from sklearn.model_selection import KFold  # for K-fold cross validation
from sklearn.model_selection import cross_val_predict  # prediction
from sklearn.model_selection import cross_val_score  # score evaluation
from sklearn.model_selection import \
    train_test_split  # training and testing data split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.naive_bayes import GaussianNB  # Naive bayes
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.neural_network import MLPClassifier
# Common Model Helpers
# Scalers
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier  # Decision Tree
from sklearn.utils import shuffle

py.init_notebook_mode(connected=True)

warnings.filterwarnings("ignore")


df = pd.read_csv('./lucid_dataset_train.csv')
y = df[['label']]
x = df.drop(['label'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# Scaling features
std_scaler = StandardScaler()
x_train = std_scaler.fit_transform(x_train)
x_test = std_scaler.transform(x_test)



# Put in our parameters for said classifiers

SEED = 0  # for reproducibility
# SEED = random.randint(0, 2**32)  # for reproducibility

print("SEED ", SEED)

configs = {
    # AdaBoost parameters
    "ada": {
        "classifier": AdaBoostClassifier,
        "params": {"n_estimators": 500, "learning_rate": 0.75, "random_state": SEED},
    },
    # Extra Trees Parameters
    "et": {
        "classifier": ExtraTreesClassifier,
        "params": {
            "n_jobs": -1,
            "n_estimators": 500,
            #'max_features': 0.5,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "verbose": 0,
            "random_state": SEED,
        },
    },
    # Gradient Boosting parameters
    "gb": {
        "classifier": GradientBoostingClassifier,
        "params": {
            "n_estimators": 500,
            #'max_features': 0.2,
            "max_depth": 5,
            "min_samples_leaf": 2,
            "verbose": 0,
            "random_state": SEED,
        },
    },
    # Random Forest parameters
    "rf": {
        "classifier": RandomForestClassifier,
        "params": {
            "n_jobs": -1,
            "n_estimators": 500,
            "warm_start": True,
            #'max_features': 0.2,
            "max_depth": 6,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "verbose": 0,
            "random_state": SEED,
        },
    },
    # Support Vector Classifier parameters
    "svc": {
        "classifier": SVC,
        "params": {"kernel": "poly", "C": 0.1, "random_state": SEED},
    },
    "dt": {
        "classifier": DecisionTreeClassifier, 
        "params": {
            "min_samples_split": 4,
            "random_state": SEED
        }
    },
}

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, config):
        self.clf = config["classifier"](**config["params"])

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)

classifiers = {}
x_train_ensemble = []
x_test_ensemble = []


for key, config in configs.items():
    clf = SklearnHelper(config)
    clf.train(x_train, y_train)
    prdn_test = clf.predict(x_test)
    prdn_train = clf.predict(x_train)
    model = {
        "classifier": clf,
        "config": config,
        "accuracy": accuracy_score(y_test, prdn_test),
        "mse": mean_squared_error(y_test, prdn_test),
        "prdn_test": prdn_test,
        "prdn_train": prdn_train,
    }

    print(key, "acc: ", model["accuracy"])
    # print(key, "(mse): ", model["mse"])
    # print()

    classifiers[key] = model
    x_train_ensemble.extend(prdn_train)
    x_test_ensemble.extend(prdn_test)


xgb_grid = GridSearchCV(
    estimator=xgb.XGBClassifier(),
    verbose=True,
    cv=5,
    scoring="roc_auc",
    n_jobs=4,
    param_grid={

        "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, .1 ],
        "n_estimators": [100, 200, 400, 800],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "gamma": [0, 1, 5],
        "subsample": [0.8, 0.85, 0.9, 0.95, 1],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "objective": ["binary:logistic"],
    },
)

xgb_grid.fit(x_train, y_train)
print(knn_grid.best_score_)
print(knn_grid.best_estimator_)



# xgboost = xgb.XGBClassifier(
#     learning_rate=0.01,
#     n_estimators=600,
#     max_depth=6,
#     gamma=6,
#     subsample=1,
#     colsample_bytree=0.95,
#     objective="binary:logistic",
# ).fit(x_train, y_train)


# predictions = xgboost.predict(x_test)

# xgb_mse = mean_squared_error(y_test, predictions)
# print("ens (mse): ", xgb_mse)

# xgb_acc = accuracy_score(y_test, predictions)
# print("xgb acc: ", xgb_acc)




knn_grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    verbose=True,
    cv=5,
    scoring="roc_auc",
    n_jobs=4,
    param_grid={
        "algorithm": ["auto"],
        "weights": ["uniform", "distance"],
        "leaf_size": list(range(1, 50, 5)),
        "n_neighbors": [5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22],
    },
)

knn_grid.fit(x_train, y_train)
print(knn_grid.best_score_)
print(knn_grid.best_estimator_)

# knn = KNeighborsClassifier(
#     algorithm="auto",
#     leaf_size=16,
#     metric="minkowski",
#     metric_params=None,
#     n_jobs=None,
#     n_neighbors=11,
#     p=2,
#     weights="uniform",
#  ).fit(x_train, y_train)

# predictions = knn.predict(x_test)

# knn_mse = mean_squared_error(y_test, predictions)
# print("knn (mse): ", ens_mse)

# knn_acc = accuracy_score(y_test, predictions)
# print("knn acc: ", ens_acc)


xgb_grid = GridSearchCV(
    estimator=xgb.XGBClassifier(),
    verbose=True,
    cv=5,
    scoring="roc_auc",
    n_jobs=4,
    param_grid={

        "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, .1 ],
        "n_estimators": [100, 200, 400, 800],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "gamma": [0, 1, 5],
        "subsample": [0.8, 0.85, 0.9, 0.95, 1],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "objective": ["binary:logistic"],
    },
)

xgb_grid.fit(x_train, y_train)
print(knn_grid.best_score_)
print(knn_grid.best_estimator_)



# xgboost = xgb.XGBClassifier(
#     learning_rate=0.01,
#     n_estimators=600,
#     max_depth=6,
#     gamma=6,
#     subsample=1,
#     colsample_bytree=0.95,
#     objective="binary:logistic",
# ).fit(x_train, y_train)


# predictions = xgboost.predict(x_test)

# xgb_mse = mean_squared_error(y_test, predictions)
# print("ens (mse): ", xgb_mse)

# xgb_acc = accuracy_score(y_test, predictions)
# print("xgb acc: ", xgb_acc)
