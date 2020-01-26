import json
import requests
import pandas as pd

df = pd.read_csv('lucid_dataset_test.csv')
random_sample = df.sample()
payload = random_sample.to_dict('records')[0]
r = requests.get('http://localhost:8080', params=payload)
print(r.content.decode("utf-8"))
