## Deploying our linear regression model on a flask application using Docker

* Create virtual environment using pipenv
* Create a prediction script
* Build an app with flask for our model deployment
* Package app using Docker


```bash
docker build -t ride-duration-prediction-service:v1 .
```

```bash
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```