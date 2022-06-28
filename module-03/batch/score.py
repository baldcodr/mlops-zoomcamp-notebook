#!/usr/bin/env python
# coding: utf-8

import pickle
import os
import sys
import pandas as pd

import mlflow
from datetime import datetime
import uuid

from prefect import task, flow
from prefect.context import get_run_context

from dateutil.relativedelta import relativedelta

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline



def generate_uuid(n):
    ride_ids = [str(uuid.uuid4()) for i in range(n)]
    return ride_ids

def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df['ride_ids'] = generate_uuid(len(df))

    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


def load_model(run_id):
    logged_model = f"s3://mlflow-models-david/1/{run_id}/artifacts/model"
    model = mlflow.pyfunc.load_model(logged_model)

    return model

def apply_model(input_file,run_id, output_file):

    print(f'reading the data from {input_file}...')
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    print(f'loading the model with RUN_ID={run_id}')
    model = load_model(run_id)

    print(f'applying the model...')
    y_pred = model.predict(dicts)

    print(f'saving the model to {output_file}...')
    df_result = pd.DataFrame()
    df_result['ride_ids'] = df['ride_ids']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['lpep_dropoff_datetime'] = df['lpep_dropoff_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id

    df_result.to_parquet(output_file, index=False)

@flow
def ride_duration_prediction(
        taxi_type: str,
        run_id: str,
        run_date: datetime = None
        ):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time

    prev_month = run_date - relativedelta(month=1)
    year = prev_month.year
    month = prev_month.month

    input_file = f'https://s3.amazonaws.com/nyc-tlc/trip+data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f's3://mlflow-models-david/taxi_type={taxi_type}/year={year:04d}/month={month:02d}/{run_id}.parquet'

    apply_model(
    input_file=input_file,
    output_file=output_file,
    run_id=run_id
    )
    

def run():
    taxi_type =sys.argv[1] #green
    year = int(sys.argv[2]) #2021
    month = int(sys.argv[3]) #3
    run_id = sys.argv[4] #61f4984eab5c4530b61689b8c512d8ec
    
    ride_duration_prediction(
        taxi_type=taxi_type,
        run_id=run_id,
        run_date=datetime(year=year, month=month, day=1)
    )


    # RUN_ID = os.getenv('RUN_ID','61f4984eab5c4530b61689b8c512d8ec')



if __name__ == '__main__':
    run()




