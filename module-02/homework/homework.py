import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
import pickle
from datetime import date
import datetime


@task
def get_paths(date):
    logger = get_run_logger()


    logger.info(f"Get paths for: {date}")
    
    # validation data
    current = date.replace(day=1)
    last_month = current - datetime.timedelta(days=1)
    val_month = last_month.month
    val_year = last_month.year
    
    # trainind data
    last_month = last_month.replace(day=1)
    dlast_month = last_month - datetime.timedelta(days=1)
    train_month = dlast_month.month
    train_year = dlast_month.year
    
    # generate the paths for validatiin and training
    train_path = f"./data/fhv_tripdata_{train_year}-{train_month:02d}.parquet"
    val_path = f"./data/fhv_tripdata_{val_year}-{val_month:02d}.parquet"
    
    logger.info(f"Train path: {train_path}")
    logger.info(f"Validation path: {val_path}")

    return train_path, val_path

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger = get_run_logger()

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
def main(date=None):

    # conditional for date input type
    if date == None:
        date = date.today()
    else:
        date = datetime.datetime.strptime(date, '%Y-%m-%d')

    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    # save our trained model
    model_path = f"./models/model-{date:%y-%m-%d}.bin"
    with open(model_path, "wb") as f_out:
            pickle.dump(lr, f_out)
    
    # save the dictionary vectorizer
    dv_path = f"./models/dv-{date:%y-%m-%d}.b"
    with open(dv_path, "wb") as f_out:
            pickle.dump(dv, f_out)

    run_model(df_val_processed, categorical, dv, lr)

main(date="2021-03-15")