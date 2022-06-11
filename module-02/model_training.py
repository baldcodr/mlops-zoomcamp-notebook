import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  

from sklearn.feature_extraction import DictVectorizer

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from sklearn.metrics import mean_squared_error
import pickle

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ny-taxi-experiment")


def read_dataframe(filename):
    df = pd.read_parquet(filename)
    #çompute trip duration in minutes
    df["duration_in_mins"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration_in_mins = df.duration_in_mins.apply(lambda td: td.total_seconds()/60)

    #Select all trips less than 60
    df = df[(df.duration_in_mins>=1) & (df.duration_in_mins <=60)]

    #Extract the most useful categorical and numerical variables from our dataset
    categorical = ['PULocationID','DOLocationID']

    #Convert numerical variables to categorical variables
    df[categorical] = df[categorical].astype(str)
    
    return df



def add_features(train_path="../module-01/data/green_tripdata_2022-01.parquet",val_path="../module-01/data/green_tripdata_2022-02.parquet"):
    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)
    
    print(df_train.shape)
    print(df_val.shape)

    #combine the pickup and drop off location ids 
    df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
    df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']

    #Extract the most useful categorical and numerical variables from our dataset
    categorical = ['PU_DO']
    numerical = ['trip_distance']

    #perform data vectorization
    dv = DictVectorizer()

    #Convert our selected features to dictionaries for our vectorizers to work
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    #Convert our selected features to dictionaries for our vectorizers to work
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)


    target = 'duration_in_mins'
    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, y_train, X_val, y_val, dv



#search for model using hyperopt
def train_model_search(train, valid, y_val):

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, "validation")],
                early_stopping_rounds=50
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}


    #Experimenting hyperparamter optimization library(hyperopt) to help us choose our model parameters for xgboost
    #Will take a while to get enough data for us to make decisions

    search_space = {
        'ma_depth': scope.int(hp.quniform('max_depth',4 , 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -1, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda':hp.loguniform('reg_lambda',-6, -1) ,
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest, 
        max_evals=1, #50 iterations
        trials=Trials()
    )

    return

#Train best model
def train_best_model(train, valid, y_val,dv):
    with mlflow.start_run():
        best_params = {
            'ma_depth': 49,
            'learning_rate': 0.4197998879005521,
            'reg_alpha': 0.02765879156310126,
            'reg_lambda': 0.30675307751450226 ,
            'min_child_weight': 2.705518581285925,
            'objective': 'reg:linear',
            'seed': 42
        }

        #log model parameters
        mlflow.log_params(best_params)

        mlflow.xgboost.autolog(disable=True)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        #log model artifacts
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, dv = add_features()
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    train_model_search(train, valid, y_val)
    train_best_model(train, valid, y_val,dv)

