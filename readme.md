# Data Science Homework Assignment

Curtis Thacker




## 1 task

Run the following commands from the root of this project.


Building the docker image:
```
docker image build -t data_science_test:0.1 .
```

Running the docker container
```
docker container run --publish 8080:8080 --detach --name ds_test data_science_test:0.1
```

Stopping the docker container
```
docker container rm --force ds_test
```
