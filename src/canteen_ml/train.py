
from pathlib import Path
import hashlib

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

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


# Project root: .../mlops-project
ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config():
    """Load project configuration from params.yaml."""
    config_path = ROOT_DIR / "params.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calculate_sha256(file_path):
    """Calculate SHA-256 hash of the original dataset."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def build_model(config):
    """Build preprocessing and regression model pipeline."""
    preprocessor = build_preprocessor()

    model_config = config["model"]

    regressor = RandomForestRegressor(
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        min_samples_split=model_config["min_samples_split"],
        min_samples_leaf=model_config["min_samples_leaf"],
        random_state=config["data"]["random_state"],
        n_jobs=-1,
    )

    from sklearn.pipeline import Pipeline

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    return model


def train():
    """Train, evaluate, and log the baseline model."""

    # 1. Load configuration
    config = load_config()

    data_path = ROOT_DIR / config["data"]["train_path"]
    target = config["data"]["target"]

    # 2. Load and validate original dataset
    df = load_dataset(data_path)
    validate_training_data(df)

    print("Original dataset shape:", df.shape)

    # 3. Clean dataset
    df = clean_training_data(df)

    print("Cleaned dataset shape:", df.shape)

    # 4. Select features and target
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = df[feature_columns]
    y = df[target]

    # 5. Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # 6. Build model
    model = build_model(config)

    # 7. Configure MLflow
    mlflow.set_experiment(
        config["training"]["experiment_name"]
    )

    # 8. Start MLflow run
    with mlflow.start_run(
        run_name="random-forest-baseline"
    ) as run:

        # Log original dataset hash INSIDE the active run
        data_hash = calculate_sha256(data_path)

        mlflow.log_param(
            "data_sha256",
            data_hash,
        )

        mlflow.set_tag(
            "model_stage",
            "baseline",
        )

        # Log dataset information
        mlflow.log_param(
            "original_rows",
            len(load_dataset(data_path)),
        )

        mlflow.log_param(
            "cleaned_rows",
            len(df),
        )

        mlflow.log_param(
            "training_rows",
            len(X_train),
        )

        mlflow.log_param(
            "validation_rows",
            len(X_val),
        )

        # 9. Train model
        model.fit(X_train, y_train)

        # 10. Predict validation data
        predictions = model.predict(X_val)

        # 11. Evaluate using MAE
        mae = mean_absolute_error(
            y_val,
            predictions,
        )

        # 12. Log model parameters
        mlflow.log_params(config["model"])

        mlflow.log_param(
            "random_state",
            config["data"]["random_state"],
        )

        # 13. Log evaluation metric
        mlflow.log_metric(
            config["training"]["metric"],
            mae,
        )

        # 14. Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        # 15. Save model locally
        models_dir = ROOT_DIR / "models"
        models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = models_dir / "model.joblib"

        joblib.dump(
            model,
            model_path,
        )

        # 16. Display results
        print("\nTraining completed successfully")
        print("--------------------------------")
        print("Original rows:", len(load_dataset(data_path)))
        print("Cleaned rows:", len(df))
        print("Training rows:", len(X_train))
        print("Validation rows:", len(X_val))
        print(f"Validation MAE: {mae:.4f}")
        print("Dataset SHA256:", data_hash)
        print("MLflow Run ID:", run.info.run_id)
        print("Saved model:", model_path)


if __name__ == "__main__":
    train()
