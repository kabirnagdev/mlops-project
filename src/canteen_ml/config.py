
from pathlib import Path

import mlflow

ROOT_DIR = Path(__file__).resolve().parents[2]
TRACKING_DB = ROOT_DIR / "mlflow.db"

def configure_mlflow():
    """Configure MLflow to use the project's local SQLite database."""
    tracking_uri = f"sqlite:///{TRACKING_DB.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri
