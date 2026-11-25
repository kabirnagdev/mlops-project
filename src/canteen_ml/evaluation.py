"""
evaluation.py — Group-wise MAE evaluation for MP033.

Owner: Kabir Nagdev (monitoring track)

Computes v1 MAE on later.csv separately for every block value
(A-Block, B-Block, Hostel) to identify the worst-served group.

Does NOT use the pre-saved model.joblib (which was pickled with a different
sklearn version). Instead it re-trains v1 on cleaned train.csv using the
same hyperparameters as params.yaml, then evaluates on later.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_validation import load_dataset  # noqa: E402
from data_cleaning import clean_training_data  # noqa: E402
from preprocessing import build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "plates_sold"
BLOCKS = ["A-Block", "B-Block", "Hostel"]

# Same hyperparameters as params.yaml
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)


def _resolve_data_path(raw_path: Path, fallback_path: Path) -> Path:
    """Return raw_path if it exists and is not a DVC stub, else fallback."""
    if raw_path.exists() and raw_path.stat().st_size > 500:
        return raw_path
    return fallback_path


def build_v1_model(train_path: Path) -> Pipeline:
    """
    Train v1: RandomForest on cleaned train.csv only.
    Returns fitted sklearn Pipeline.
    """
    df_raw = load_dataset(train_path)
    df = clean_training_data(df_raw)

    X = df[FEATURE_COLS]
    y = df[TARGET]

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(**RF_PARAMS)),
    ])
    pipeline.fit(X, y)
    print(f"  v1 trained on {len(X)} rows from {train_path.name}")
    return pipeline


def compute_mae_by_block(
    model: Pipeline,
    later_path: Path,
) -> dict[str, float]:
    """
    Compute MAE for model on later.csv, split by block.

    Parameters
    ----------
    model      : Fitted sklearn Pipeline.
    later_path : Path to later.csv.

    Returns
    -------
    dict mapping block name → MAE (float).
    """
    df_later = load_dataset(later_path)
    block_maes: dict[str, float] = {}

    for block in BLOCKS:
        subset = df_later[df_later["block"] == block].copy()

        if subset.empty:
            print(f"  WARNING: No rows for block '{block}' in later.csv")
            block_maes[block] = float("nan")
            continue

        X_block = subset[FEATURE_COLS]
        y_block = subset[TARGET]

        preds = model.predict(X_block)
        mae = mean_absolute_error(y_block, preds)
        block_maes[block] = float(round(mae, 4))
        print(f"  MAE [{block}]: {mae:.4f}  (n={len(subset)})")

    return block_maes


def identify_worst_served(block_maes: dict[str, float]) -> str:
    """Return the block name with the highest MAE."""
    valid = {k: v for k, v in block_maes.items() if not np.isnan(v)}
    return max(valid, key=valid.get)


def run_group_evaluation(
    train_path: Path | None = None,
    later_path: Path | None = None,
) -> tuple[Pipeline, dict[str, float], str]:
    """
    Full group evaluation pipeline.

    Returns
    -------
    (v1_model, mae_by_block, worst_block)
    """
    if train_path is None:
        raw_train = ROOT_DIR / "data" / "raw" / "train.csv"
        train_path = _resolve_data_path(raw_train, ROOT_DIR / "data" / "train.csv")
    if later_path is None:
        raw_later = ROOT_DIR / "data" / "raw" / "later.csv"
        later_path = _resolve_data_path(raw_later, ROOT_DIR / "data" / "later.csv")

    print("Building v1 model...")
    v1 = build_v1_model(train_path)

    print("\nComputing MAE by block on later.csv...")
    block_maes = compute_mae_by_block(v1, later_path)

    worst = identify_worst_served(block_maes)
    print(f"\nWorst-served group: {worst} (MAE = {block_maes[worst]:.4f})")

    return v1, block_maes, worst


if __name__ == "__main__":
    print("=" * 60)
    print("Group-wise MAE Evaluation — MP033")
    print("=" * 60)
    v1_model, maes, worst_block = run_group_evaluation()
    print("\nSummary:")
    for b, mae in maes.items():
        marker = " <- WORST" if b == worst_block else ""
        print(f"  {b:<10s}: MAE = {mae:.4f}{marker}")
