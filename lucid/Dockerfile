FROM python:3.7.6
LABEL maintainer "curtis.thacker@gmail.com"
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY simple_api.py simple_api.py
COPY model.pkl model.pkl
COPY model_variables.json model_variables.json
EXPOSE 8080 
CMD python simple_api.py