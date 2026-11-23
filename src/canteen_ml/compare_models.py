
from pathlib import Path
import hashlib

import mlflow
import mlflow.sklearn
import yaml

from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_validation import (
    load_dataset,
    validate_training_data,
)
from data_cleaning import clean_training_data
from preprocessing import (
    build_preprocessor,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT_DIR / "params.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calculate_sha256(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def build_pipeline(regressor):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", regressor),
    ])


def compare_models():
    config = load_config()

    data_path = ROOT_DIR / config["data"]["train_path"]
    target = config["data"]["target"]

    df = load_dataset(data_path)
    validate_training_data(df)
    df = clean_training_data(df)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = df[features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),

        "decision_tree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=3,
            random_state=42,
        ),
    }

    mlflow.set_experiment(
        config["training"]["experiment_name"]
    )

    dataset_hash = calculate_sha256(data_path)
    results = []

    for model_name, regressor in models.items():

        pipeline = build_pipeline(regressor)

        with mlflow.start_run(run_name=model_name) as run:

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_val)

            mae = mean_absolute_error(
                y_val,
                predictions,
            )

            mlflow.log_param("algorithm", model_name)
            mlflow.log_param("data_sha256", dataset_hash)
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param("validation_rows", len(X_val))

            mlflow.log_metric("mae", mae)

            mlflow.set_tag("model_stage", "comparison")
            mlflow.set_tag("validation_method", "random_holdout")

            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            results.append({
                "model": model_name,
                "mae": mae,
                "run_id": run.info.run_id,
            })

            print(f"\nModel: {model_name}")
            print(f"MAE: {mae:.4f}")
            print(f"Run ID: {run.info.run_id}")

    results.sort(key=lambda item: item["mae"])

    print("\n========== MODEL COMPARISON ==========")

    for result in results:
        print(
            f"{result['model']:20s} "
            f"MAE: {result['mae']:.4f} "
            f"Run ID: {result['run_id']}"
        )

    print("\nBest model:", results[0]["model"])
    print("Best MAE:", results[0]["mae"])


if __name__ == "__main__":
    compare_models()
