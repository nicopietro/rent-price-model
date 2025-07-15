import os
import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.data import from_pandas
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os

load_dotenv()

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)

def check_mlflow_artifacts_bucket(bucket_name="mlflow-artifacts") -> None:
    """Check if the MLflow artifacts bucket exists, create it if not."""
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            s3.create_bucket(Bucket=bucket_name)

check_mlflow_artifacts_bucket()

mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment("rent-price-prediction")

def save_to_mlflow(
        model: Pipeline,
        model_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        ) -> None:
    """Saves the model and metrics to MLflow."""

    y_pred = best_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    signature = infer_signature(X_train, y_pred)

    with mlflow.start_run(run_name=model_name) as run:

        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('R2', r2)
        mlflow.log_params(model.get_params())
        mlflow.log_input(from_pandas(X_train), "X_train")
        mlflow.log_input(from_pandas(X_test), "X_test")
        mlflow.log_input(from_pandas(pd.DataFrame({'target': y_train})), "y_train")
        mlflow.log_input(from_pandas(pd.DataFrame({'target': y_test})), "y_test")


        # Log the pipeline (model + preprocessor) as a single MLflow artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            signature=signature,
            input_example=X_train.sample(5, random_state=42),
            registered_model_name=model_name,
        )

        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")


df = pd.read_csv('./data/cleaned/rent_dataset_clean.csv')

target_col = 'Rent'
feature_cols = [col for col in df.columns if col != target_col]
numeric_cols = ['Size', 'BHK', 'Bathroom']
categorical_cols = [col for col in feature_cols if col not in numeric_cols]
print(f'Feature columns: {feature_cols}')

X = df[feature_cols]
y = df[target_col].to_numpy().flatten()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

param_grids = {
    'Linear Regression': {},
    'Random Forest': {
        'model__n_estimators': [50, 100],
        'model__max_depth': [None, 10],
    },
    'XGBoost': {
        'model__n_estimators': [50, 100],
        'model__max_depth': [3, 6],
        'model__learning_rate': [0.1, 0.3],
    },
    'LightGBM': {
        'model__n_estimators': [50, 100],
        'model__max_depth': [5, 10],
        'model__learning_rate': [0.1, 0.3],
    },
}

models = {
    'Linear Regression': Pipeline([
        ('preprocess', preprocessor),
        ('model', LinearRegression())
    ]),
    'Random Forest': Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('preprocess', preprocessor),
        ('model', XGBRegressor(random_state=42))
    ]),
    'LightGBM': Pipeline([
        ('preprocess', preprocessor),
        ('model', LGBMRegressor(random_state=42))
    ]),
}


for name, pipeline in models.items():
    print(f"Running Grid Search for: {name}")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[name],
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    save_to_mlflow(
        model=best_model,
        model_name=name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )