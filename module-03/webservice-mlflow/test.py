import requests

ride = {
    "PULocationID":460,
    "DOLocationID":500,
    "trip_distance":700,
}
url = "http://localhost:9696/predict"
response = requests.post(url, json=ride)
print(response.json())