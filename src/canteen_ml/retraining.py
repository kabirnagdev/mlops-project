"""
retraining.py — Temporal v1 vs v2 retraining and promotion logic for MP033.

Owner: Kabir Nagdev (monitoring track)

Procedure
---------
1. Sort later.csv by date.
2. First 70% of later.csv → retraining split.
3. Last  30% of later.csv → check set (same for both v1 and v2).
4. v2 = RandomForest trained on cleaned train.csv + first 70% of later.csv.
5. Evaluate v1 and v2 MAE on the identical last 30% check set.
6. Promotion rule: promote v2 only if improvement >= 12%.

   improvement_percent = ((MAE_v1 - MAE_v2) / MAE_v1) * 100

7. Save v2 model to models/model_v2.joblib.
8. If promoted, overwrite models/model.joblib with v2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
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
PROMOTION_THRESHOLD_PCT = 12.0  # promote v2 only if >= 12% better

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)


def _resolve_path(raw: Path, fallback: Path) -> Path:
    return raw if (raw.exists() and raw.stat().st_size > 500) else fallback


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(**RF_PARAMS)),
    ])


def _split_later_temporal(later_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort later.csv by date and split 70/30.

    Returns
    -------
    (retrain_split, check_set)
    """
    df = load_dataset(later_path)
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * 0.70)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def run_temporal_retraining(
    train_path: Path | None = None,
    later_path: Path | None = None,
    v1_model: Pipeline | None = None,
) -> dict:
    """
    Execute the temporal v1 vs v2 evaluation and promotion decision.

    Parameters
    ----------
    train_path : Path to train.csv.
    later_path : Path to later.csv.
    v1_model   : Already-fitted v1 Pipeline. If None, v1 is retrained here.

    Returns
    -------
    dict with keys:
        retrain_rows, check_rows,
        mae_v1, mae_v2,
        improvement_pct, decision,
        champion  ('v1' or 'v2')
    """
    if train_path is None:
        train_path = _resolve_path(
            ROOT_DIR / "data" / "raw" / "train.csv",
            ROOT_DIR / "data" / "train.csv",
        )
    if later_path is None:
        later_path = _resolve_path(
            ROOT_DIR / "data" / "raw" / "later.csv",
            ROOT_DIR / "data" / "later.csv",
        )

    # ---- 1. Load & clean training data ----
    df_train_raw = load_dataset(train_path)
    df_train = clean_training_data(df_train_raw)

    # ---- 2. Temporal split of later.csv ----
    retrain_split, check_set = _split_later_temporal(later_path)
    print(f"  later.csv split: retrain={len(retrain_split)} rows | check={len(check_set)} rows")

    # ---- 3. Prepare check-set features / labels ----
    X_check = check_set[FEATURE_COLS]
    y_check = check_set[TARGET]

    # ---- 4. Build / use v1 ----
    if v1_model is None:
        print("  Training v1 on cleaned train.csv only...")
        v1_model = _build_pipeline()
        v1_model.fit(df_train[FEATURE_COLS], df_train[TARGET])

    # ---- 5. Evaluate v1 on check set ----
    preds_v1 = v1_model.predict(X_check)
    mae_v1 = float(round(mean_absolute_error(y_check, preds_v1), 4))
    print(f"  v1 MAE on check set: {mae_v1:.4f}")

    # ---- 6. Build v2: train.csv + retrain_split ----
    print("  Training v2 on cleaned train.csv + first 70% of later.csv...")
    df_v2 = pd.concat(
        [df_train, retrain_split],
        ignore_index=True,
    )
    # later.csv is already clean — no sentinel / duplicate issues per PROBLEM.md
    X_v2_train = df_v2[FEATURE_COLS]
    y_v2_train = df_v2[TARGET]

    v2_model = _build_pipeline()
    v2_model.fit(X_v2_train, y_v2_train)
    print(f"  v2 trained on {len(X_v2_train)} rows total")

    # ---- 7. Evaluate v2 on the same check set ----
    preds_v2 = v2_model.predict(X_check)
    mae_v2 = float(round(mean_absolute_error(y_check, preds_v2), 4))
    print(f"  v2 MAE on check set: {mae_v2:.4f}")

    # ---- 8. Promotion decision ----
    improvement_pct = ((mae_v1 - mae_v2) / mae_v1) * 100
    print(f"  Improvement: {improvement_pct:.2f}%  (threshold: {PROMOTION_THRESHOLD_PCT}%)")

    if improvement_pct >= PROMOTION_THRESHOLD_PCT:
        decision = f"promote_v2 (improvement {improvement_pct:.2f}% >= {PROMOTION_THRESHOLD_PCT}%)"
        champion = "v2"
        champion_model = v2_model
        print(f"  Decision: PROMOTE v2 — {decision}")
    else:
        decision = f"retain_v1 (improvement {improvement_pct:.2f}% < {PROMOTION_THRESHOLD_PCT}%)"
        champion = "v1"
        champion_model = v1_model
        print(f"  Decision: RETAIN v1 — {decision}")

    # ---- 9. Save models ----
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Always save v1 (for auditing)
    v1_path = models_dir / "model_v1.joblib"
    joblib.dump(v1_model, v1_path)
    print(f"  Saved v1 -> {v1_path}")

    # Always save v2
    v2_path = models_dir / "model_v2.joblib"
    joblib.dump(v2_model, v2_path)
    print(f"  Saved v2 -> {v2_path}")

    # Save champion
    champ_path = models_dir / "model.joblib"
    joblib.dump(champion_model, champ_path)
    print(f"  Saved champion ({champion}) -> {champ_path}")

    return {
        "retrain_rows": len(retrain_split),
        "check_rows": len(check_set),
        "mae_v1": mae_v1,
        "mae_v2": mae_v2,
        "improvement_pct": round(improvement_pct, 4),
        "decision": decision,
        "champion": champion,
        "v1_model": v1_model,
        "v2_model": v2_model,
        "champion_model": champion_model,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Temporal Retraining & Promotion — MP033")
    print("=" * 60)
    result = run_temporal_retraining()
    print("\nResult summary:")
    for k, v in result.items():
        if k not in ("v1_model", "v2_model", "champion_model"):
            print(f"  {k}: {v}")
