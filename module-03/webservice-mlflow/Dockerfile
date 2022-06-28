FROM python:3.9

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy
RUN export RUN_ID="61f4984eab5c4530b61689b8c512d8ec"

COPY ["predict.py", "./"]

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]