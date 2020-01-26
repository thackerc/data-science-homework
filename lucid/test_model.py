import json
import joblib as jl
import pandas as pd

# Read in data source
test_df = pd.read_csv('lucid_dataset_test.csv')
x = test_df

# Read in variables
with open('model_variables.json', 'r') as f:
    model_variables = json.load(f)

column_names = model_variables['column_names']

# Clean features
curr_column_names = list(x.columns)

# Fix column names and order
for col in column_names:
    if not (col in curr_column_names):
        x[col] = 0

drop_list = []
for curr_col in curr_column_names:
    if not (curr_col in column_names):
        drop_list.append(curr_col)
x = x.drop(drop_list, axis=1)
x = x[column_names]

# Predict results
model = jl.load('model.pkl')
y = model.predict(x.to_numpy())
print(y)
