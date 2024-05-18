FROM python:3.8-slim-buster
WORKDIR /sevice
COPY requirements.txt .

COPY . ./

RUN  pip install -r requirements.txt
CMD ["python3", "app.py"]