# Data Science Homework Assignment

Curtis Thacker

## The New Model

A simple classifier for the lucid dataset can be found in the [lucid](./lucid) folder in this repo. It was developed using Python 3.7.6. The pickled version of the trained model is in the same directory. Accuracy: `0.974237190558434`. Mean Squared Error: `0.025762809441565917`

* The lucid dataset is all continious data; so no dummies were needed and none of the columns needed to be dropped.
* The labels don't give any information about how the data could be enhanced based on domain specific knowledge.
* There were no missing values to address.
* XGBoost is a good choice for this type of problem as it does all the bagging, boosting, pruning and normalization of the data for you. Particulary useful when the data set is annonymous.
* The XGBoost model parameters were tuned and several other model types were tested.

## Updating and Adapting the Old Model

* The Dockerfiles now reference python:3.7.6. [titanic](./titanic/Dockerfile), [lucid](./lucid/Dockerfile)
* Several dependencies had to be updated to be compatible with python 3.
    * SocketServer is now socketserver
    * urlparse is now url.parse
    * BaseHTTPServer is now http.server
* The requirements file was update to use current versions of the dependancies.
* The API accepts JSON data via POST as a method of processing batch data.
* [./lucid/batch_request.py](./lucid/batch_request.py) and [./titanic/batch_request.py](./titanic/batch_request.py) were added to test the ability to post a batch of data for processing.
* GET and POST responses are now in JSON.

NOTE: A production system would also implement parameter validation, error handling, logging, and security.


## Running the New Model

The lucid model can be run using these commands on the commandline. Start by navigating to the [lucid](./lucid) directory.

Building the docker image
```
docker image build -t data_science_test:0.1 .
```

Running the docker container
```
docker container run --publish 8080:8080 --detach --name ds_test data_science_test:0.1
```

Testing a simple request
```
python3 simple_request.py
```

Testing a batch request
```
python3 batch_request.py
```

Stopping the docker container
```
docker container rm --force ds_test
```