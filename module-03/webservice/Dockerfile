FROM python:3.10-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "linear_reg.bin", "./"]

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]