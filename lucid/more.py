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
    cv=5,
    scoring="roc_auc",
    n_jobs=4,
    param_grid={

        "learning_rate": [0.01, 0.02, 0.03, 0.05,  0.07, .1 ],
        "n_estimators": [100, 300, 800],
        "max_depth": [3, 4, 5],
        "gamma": [0, 1, 5],
        "subsample": [0.8, 0.9, 1],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.8],
        "objective": ["binary:logistic"],
    },
)

xgb_grid.fit(x, y.values.ravel())
print(xgb_grid.best_score_)
print(xgb_grid.best_estimator_)



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
