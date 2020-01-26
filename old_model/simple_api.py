import json
import xgboost
import urllib.parse as urlparse
import socketserver
import pandas as pd
import joblib as jl
from http.server import BaseHTTPRequestHandler

def convert_query(query):
    """Converts a query string into a Pandas DataFrame for a single instance of a dataset

    Args:
        query (string): an HTTP query string with key value pairs. E.g., "foo=bar&bin=bash"
    """
    columns = [col.split('=')[0] for col in query.split('&')]
    values = [val.split('=')[1] for val in query.split('&')]
    input_df = pd.DataFrame([values], columns=columns)
    return input_df


def run_model(test_df):
    """Returns predictions from a pretrained model. The model is loaded from a pkl file.
    Data is prepares the test data the same way it was prepared in training.

    Args:
        test_df (pd.DataFrame): The test data used to create the prediction.
    """
    passenger_ids = test_df['PassengerId']
    x = test_df.drop(['Name', 'PassengerId'], axis=1)
    
    # Read in variables
    with open('model_variables.json', 'r') as f:
        model_variables = json.load(f)

    age = model_variables['age']
    column_names = model_variables['column_names']

    # Clean features
    gender_dict = {'male': 0, 'female': 1}
    x['Sex'] = x['Sex'].map(gender_dict) 
    x['Age'] = x['Age'].fillna(age)
    x['Cabin'] = x['Cabin'].fillna('Unknown')
    x = pd.get_dummies(x, 
                       prefix=['Ticket', 'Cabin', 'Embarked'], 
                       columns=['Ticket', 'Cabin', 'Embarked']
                      )
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
    
    return pd.DataFrame({
        "PassengerId": passenger_ids, 
        "Survived": y
    })


class MyHandler(BaseHTTPRequestHandler):

    def _headers(self):
        """Sets the headers for responses. This is not the way to do this in production.
        """
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    def do_GET(self):
        """Handles GET HTTP Requests.
        Uses the query string to create an instance.
        """

        self._headers()

        path = urlparse.urlparse(self.path)
        input_df = convert_query(path.query)
        result = run_model(input_df)
        self.wfile.write(result.to_json(orient='records').encode())

    def do_POST(self):
        """Handles GET HTTP Requests.
        Assumes the body of the raw data posted in the request is JSON. That JSON is parsed into 
        a batch of instances to be used for prediction.
        """
        self._headers()

        json_data_string = self.rfile.read(int(self.headers['Content-Length'])).decode("utf-8")
        input_df = pd.DataFrame(json.loads(json_data_string))
        result = run_model(input_df)
        self.wfile.write(result.to_json(orient='records').encode())

if __name__ == "__main__":
    httpd = socketserver.TCPServer(("", 8080), MyHandler)
    httpd.serve_forever()
