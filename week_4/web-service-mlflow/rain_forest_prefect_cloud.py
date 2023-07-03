import pickle
import numpy as np
import scipy
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline
from datetime import date
import pathlib
import mlflow

from prefect import flow, task
from prefect_aws import S3Bucket
from prefect.artifacts import create_markdown_artifact
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("green-taxi-duration")


@task(retries=3, retry_delay_seconds=2)
def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df

@task(retries=3, retry_delay_seconds=2)
def prepare_dictionaries(df: pd.DataFrame):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts

@task(log_prints=True)
def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix,
    X_val: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    # dv: sklearn.feature_extraction.DictVectorizer,
) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run():
        # train = xgb.DMatrix(X_train, label=y_train)
        # valid = xgb.DMatrix(X_val, label=y_val)

        # best_params = {
        #     "learning_rate": 0.09585355369315604,
        #     "max_depth": 30,
        #     "min_child_weight": 1.060597050922164,
        #     "objective": "reg:linear",
        #     "reg_alpha": 0.018060244040060163,
        #     "reg_lambda": 0.011658731377413597,
        #     "seed": 42,
        # }

        # mlflow.log_params(best_params)

        # booster = xgb.train(
        #     params=best_params,
        #     dtrain=train,
        #     num_boost_round=100,
        #     evals=[(valid, "validation")],
        #     early_stopping_rounds=20,
        # )

        # y_pred = booster.predict(valid)
        # rmse = mean_squared_error(y_val, y_pred, squared=False)
        # mlflow.log_metric("rmse", rmse)

        params = dict(max_depth=20, n_estimators=100, min_samples_leaf=10, random_state=0)
        mlflow.log_params(params)

        pipeline = make_pipeline(
            DictVectorizer(),
            RandomForestRegressor(**params, n_jobs=-1)
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        rmse = mean_squared_error(y_pred, y_val, squared=False)
        print(params, rmse)
        mlflow.log_metric('rmse', rmse)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        pathlib.Path("models").mkdir(exist_ok=True)
        # with open("models/preprocessor.b", "wb") as f_out:
        #     pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        # mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        mlflow.sklearn.log_model(pipeline, artifact_path="models")

        markdown__rmse_report = f"""# RMSE Report
            ## Summary

            Duration Prediction 

            ## RMSE XGBoost Model

            | Region    | RMSE |
            |:----------|-------:|
            | {date.today()} | {rmse:.2f} |
            """

        create_markdown_artifact(
            key="duration-model-report", markdown=markdown__rmse_report
        )

    return None


# df_train = read_dataframe('../../dataset/green_tripdata_2021-01.parquet')
# df_val = read_dataframe('../../dataset/green_tripdata_2021-02.parquet')

# target = 'duration'
# y_train = df_train[target].values
# y_val = df_val[target].values

# dict_train = prepare_dictionaries(df_train)
# dict_val = prepare_dictionaries(df_val)

# with mlflow.start_run():
#     params = dict(max_depth=20, n_estimators=100, min_samples_leaf=10, random_state=0)
#     mlflow.log_params(params)

#     pipeline = make_pipeline(
#         DictVectorizer(),
#         RandomForestRegressor(**params, n_jobs=-1)
#     )

#     pipeline.fit(dict_train, y_train)
#     y_pred = pipeline.predict(dict_val)

#     rmse = mean_squared_error(y_pred, y_val, squared=False)
#     print(params, rmse)
#     mlflow.log_metric('rmse', rmse)

#     mlflow.sklearn.log_model(pipeline, artifact_path="model")

@flow
def main_flow_week_4_s3(
    train_path: str = "dataset/green_tripdata_2021-01.parquet",
    val_path: str = "dataset/green_tripdata_2021-02.parquet",
) -> None:
    """The main training pipeline"""

    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

    # Load
    s3_bucket_block = S3Bucket.load("s3-bucket-example")
    s3_bucket_block.download_folder_to_path(from_folder="dataset", to_folder="dataset")
    s3_bucket_block.download_folder_to_path(from_folder="models", to_folder="models")

    # df_train = read_data(train_path)
    # df_val = read_data(val_path)

    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)


    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    dict_train = prepare_dictionaries(df_train)
    dict_val = prepare_dictionaries(df_val)

    # Transform
    # X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Train
    train_best_model(dict_train, dict_val, y_train, y_val)


if __name__ == "__main__":
    main_flow_week_4_s3()