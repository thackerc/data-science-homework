import json
import requests
import pandas as pd

df = pd.read_csv('test.csv')
random_sample = df.sample(10)
payload = random_sample.to_dict('records')
r = requests.post('http://localhost:8080', json=payload)
print(r.content.decode("utf-8"))