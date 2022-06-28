## Deploying our random forest model on a flask application using Docker

* Create virtual environment using pipenv
* Create a prediction script
* Build an app with flask for our model deployment
* Package app using Docker


```bash
docker build -t ride-duration-prediction-service:v2 .
```

```bash
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v2
```

docker run -it --rm -p 9696:9696 -e AWS_ACCESS_KEY_ID=XXXX -e AWS_SECRET_ACCESS_KEY=XXXXX  ride-duration-prediction-service:v2 --env RUN_ID="61f4984eab5c4530b61689b8c512d8ec"