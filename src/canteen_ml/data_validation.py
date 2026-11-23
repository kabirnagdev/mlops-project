
from pathlib import Path

import pandas as pd


REQUIRED_FEATURES = [
    "students_on_campus",
    "menu_price_rs",
    "temp_c",
    "exam_days_left",
    "special_dishes",
    "queue_minutes_yesterday",
    "day_type",
    "block",
]

TARGET = "plates_sold"


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset without modifying the source file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def validate_training_data(df: pd.DataFrame) -> None:
    """Validate the required training columns and basic data integrity."""
    required = set(REQUIRED_FEATURES + [TARGET])
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Training dataset is empty.")

    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")

    if (df[TARGET] < 0).any():
        raise ValueError("Target contains negative values.")


def validate_prediction_data(df: pd.DataFrame) -> None:
    """Validate that prediction input contains all required features."""
    missing = set(REQUIRED_FEATURES).difference(df.columns)

    if missing:
        raise ValueError(f"Missing prediction features: {sorted(missing)}")

    if df.empty:
        raise ValueError("Prediction dataset is empty.")
