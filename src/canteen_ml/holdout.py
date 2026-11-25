"""
holdout.py — Generate holdout predictions for MP033.

Owner: Kabir Nagdev (monitoring track)

Loads the champion model and predicts plates_sold for every row
in data/raw/holdout_features.csv.

Output: predictions/holdout_predictions.csv
Columns: record_id, prediction (exactly two columns, no extra index).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXPECTED_ROWS = 250


def _resolve_path(raw: Path, fallback: Path) -> Path:
    return raw if (raw.exists() and raw.stat().st_size > 500) else fallback


def generate_holdout_predictions(
    holdout_path: Path | None = None,
    model_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Generate holdout predictions using the champion model.

    Parameters
    ----------
    holdout_path : Path to holdout_features.csv.
    model_path   : Path to champion model.joblib.
    output_path  : Where to write holdout_predictions.csv.

    Returns
    -------
    DataFrame with columns ['record_id', 'prediction'].
    """
    if holdout_path is None:
        holdout_path = _resolve_path(
            ROOT_DIR / "data" / "raw" / "holdout_features.csv",
            ROOT_DIR / "data" / "holdout_features.csv",
        )
    if model_path is None:
        model_path = ROOT_DIR / "models" / "model.joblib"
    if output_path is None:
        output_path = ROOT_DIR / "predictions" / "holdout_predictions.csv"

    # ---- 1. Load holdout features ----
    df = pd.read_csv(holdout_path)
    print(f"  Holdout features loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # ---- 2. Validate row count ----
    if df.shape[0] != EXPECTED_ROWS:
        print(
            f"  WARNING: Expected {EXPECTED_ROWS} rows, got {df.shape[0]}. "
            "Continuing — marking tool uses actual file."
        )

    # ---- 3. Validate no plates_sold column present ----
    if "plates_sold" in df.columns:
        raise ValueError(
            "holdout_features.csv must NOT contain 'plates_sold'. "
            "Do not fabricate holdout labels."
        )

    # ---- 4. Load champion model ----
    if not model_path.exists():
        raise FileNotFoundError(
            f"Champion model not found at {model_path}. "
            "Run monitoring.py first to train and save the champion."
        )
    model = joblib.load(model_path)
    print(f"  Champion model loaded from {model_path.name}")

    # ---- 5. Generate predictions ----
    X = df[FEATURE_COLS]
    preds = model.predict(X)

    # ---- 6. Build output DataFrame ----
    out = pd.DataFrame({
        "record_id": df["record_id"].values,
        "prediction": preds,
    })

    # ---- 7. Validate output ----
    assert len(out) == len(df), "Row count mismatch in output"
    assert out["prediction"].notna().all(), "Missing predictions detected"
    assert list(out.columns) == ["record_id", "prediction"], "Wrong output columns"

    # ---- 8. Save CSV ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"  Holdout predictions saved -> {output_path}")
    print(f"  Rows: {len(out)} | Columns: {list(out.columns)}")
    print(f"  Prediction range: [{preds.min():.2f}, {preds.max():.2f}]")

    return out


if __name__ == "__main__":
    print("=" * 60)
    print("Holdout Prediction Generation — MP033")
    print("=" * 60)
    df_preds = generate_holdout_predictions()
    print("\nFirst 5 rows:")
    print(df_preds.head())
    print("\nLast 5 rows:")
    print(df_preds.tail())
