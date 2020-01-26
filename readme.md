# Data Science Homework Assignment

Curtis Thacker

## The New Model

A simple classifier for the lucid dataset can be found in the lucid folder in this repo. It was developed using Python 3.7.6. The pickled version of the trained model is in the same directory.

* Accuracy: `0.974237190558434`
* Mean Squared Error: `0.025762809441565917`

Thoughts:
* The lucid dataset is all continious data so no dummies were needed and none of the columns needed to be dropped.
* The labels don't give any information about how the data could be enhanced based on domain specific knowledge.
* There were no missing values to address.
* I tried perform standardization by centering and scaling all values, but this didn't affect the accuracy of the model.

## Updating and Adapting the Old Model

* The Dockerfiles now reference python:3.7.6. [titanic](./titanic/Dockerfile), [lucid](./lucid/Dockerfile)
* Several dependancies had to be update to be compatible with python 3.
    * SocketServer is now socketserver
    * urlparse is now in url.parse
    * BaseHTTPServer is now in http.server
* Also update the requirements file to use more current version of the dependancies referenced there.
* The API accepts JSON data via POST as a method of processing batch data.
* [./lucid/batch_request.py](./lucid/batch_request.py) and [./titanic/batch_request.py](./titanic/batch_request.py) were added to test the ability to post a batch of data for processing.
* GET and POST responses are now in JSON.

NOTE: A production system would also implement parameter validation, error handling, logging, and security.


## Running the New Model

The lucid model can be run using these commands on the commandline. Start by navigating to the `lucid` directory.

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