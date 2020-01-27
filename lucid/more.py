import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


df = pd.read_csv('./lucid_dataset_train.csv')
y = df[['label']]
x = df.drop(['label'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)



xgb_grid = GridSearchCV(
    estimator=xgb.XGBClassifier(),
    verbose=True,
    cv=3,
    scoring="roc_auc",
    n_jobs=4,
    param_grid={

        "learning_rate": [0.07,  0.08, 0.09 ],
        "n_estimators": [ 400 ],
        "max_depth": [3, 4],
        "gamma": [0, 1, 5],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [ 0.4, 0.5, 0.6, 0.8],
        "objective": ["binary:logistic"],
    },
)

xgb_grid.fit(x, y.values.ravel())
print(xgb_grid.best_score_)
print(xgb_grid.best_estimator_)
