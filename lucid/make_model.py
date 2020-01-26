import json
import joblib as jl
import pandas as pd
import xgboost as xgb

from sklearn.metrics import accuracy_score, mean_squared_error

# Read in data source and split label from features
df = pd.read_csv('./lucid_dataset_train.csv')
y = df[['label']]
x = df.drop(['label'], axis=1)


print(x.info())
print(x.shape)
print(x.describe())


# Build model
model = xgb.XGBClassifier()
model.fit(x.to_numpy(), y.values.ravel())

# Evaluate the accuracy and error of the model
# Cross validation would be better here.
predictions = model.predict(x.to_numpy())
mse = mean_squared_error(y, predictions)
print("mse: ", mse)

acc = accuracy_score(y, predictions)
print("acc: ", acc)

# Save model
model_file = 'model.pkl'
jl.dump(model, model_file)
column_names = list(x.columns)
model_variables = { 'column_names': column_names }
with open('model_variables.json', 'w') as f:
    json.dump(model_variables, f)

