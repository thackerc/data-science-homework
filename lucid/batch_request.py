import json
import requests
import pandas as pd

df = pd.read_csv('lucid_dataset_test.csv')
random_sample = df.sample(10)
payload = random_sample.to_dict('records')
r = requests.post('http://localhost:8080', json=payload)
response = json.loads(r.content.decode("utf-8"))
print(json.dumps(response, indent=4, separators=(',', ': ')))
