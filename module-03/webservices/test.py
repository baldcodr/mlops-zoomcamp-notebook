import requests

ride = {
    "PULocationID":100,
    "DOLocationID":500000,
    "trip_distance":500,
}
url = "http://localhost:9696/predict"
response = requests.post(url, json=ride)
print(response.json())