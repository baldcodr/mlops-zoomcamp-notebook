import os
import pickle
from flask import Flask, request, jsonify

import mlflow

# from mlflow.tracking import MlflowClient
# MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# logged_model = f'runs:/{RUN_ID}/model'


# RUN_ID = '61f4984eab5c4530b61689b8c512d8ec'

RUN_ID = os.getenv('RUN_ID')
logged_model = f"s3://mlflow-models-david/1/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'],ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    preds = model.predict(features)
    return preds[0]


app = Flask('duration_prediction')

@app.route('/predict',methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host = 'localhost', port =9696)