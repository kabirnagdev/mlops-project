"""
drift.py — PSI (Population Stability Index) calculation for MP033.

Owner: Kabir Nagdev (monitoring track)

Computes PSI between cleaned train.csv and later.csv for every numeric feature.
Uses 10 bins with consistent edges derived from the reference (train) distribution.
Handles zero proportions safely to avoid log-zero / division-by-zero errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_validation import load_dataset  # noqa: E402
from data_cleaning import clean_training_data  # noqa: E402

NUMERIC_FEATURES = [
    "students_on_campus",
    "menu_price_rs",
    "temp_c",
    "exam_days_left",
    "special_dishes",
    "queue_minutes_yesterday",
]

N_BINS = 10
EPSILON = 1e-6  # small floor to avoid log(0)


# ---------------------------------------------------------------------------
# Core PSI helpers
# ---------------------------------------------------------------------------

def _safe_proportions(counts: np.ndarray) -> np.ndarray:
    """Convert bin counts to proportions, flooring at EPSILON to avoid log-zero."""
    total = counts.sum()
    props = counts / total if total > 0 else np.ones(len(counts)) / len(counts)
    return np.where(props == 0, EPSILON, props)


def compute_psi_for_feature(
    reference: pd.Series,
    comparison: pd.Series,
    n_bins: int = N_BINS,
) -> float:
    """
    Compute Population Stability Index for a single numeric feature.

    Parameters
    ----------
    reference  : Series from the reference dataset (cleaned train).
    comparison : Series from the comparison dataset (later.csv).
    n_bins     : Number of equal-width bins (default 10).

    Returns
    -------
    float — PSI value.

    PSI interpretation
    ------------------
    PSI < 0.10  — no significant change
    PSI 0.10-0.20 — moderate shift (monitor)
    PSI > 0.20  — significant drift (investigate / retrain)
    """
    ref_clean = reference.dropna()
    cmp_clean = comparison.dropna()

    # Build bin edges from the reference distribution
    min_val = ref_clean.min()
    max_val = ref_clean.max()

    # Extend edges slightly so max value falls inside last bin
    bin_edges = np.linspace(min_val, max_val + EPSILON, n_bins + 1)

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cmp_counts, _ = np.histogram(cmp_clean, bins=bin_edges)

    ref_props = _safe_proportions(ref_counts.astype(float))
    cmp_props = _safe_proportions(cmp_counts.astype(float))

    psi = np.sum((cmp_props - ref_props) * np.log(cmp_props / ref_props))
    return float(round(psi, 6))


def compute_all_psi(
    train_path: Path | None = None,
    later_path: Path | None = None,
) -> dict[str, float]:
    """
    Compute PSI for all numeric features.

    Parameters
    ----------
    train_path : Path to train.csv (defaults to data/raw/train.csv).
    later_path : Path to later.csv (defaults to data/raw/later.csv).

    Returns
    -------
    dict mapping feature name → PSI value.
    """
    if train_path is None:
        train_path = ROOT_DIR / "data" / "raw" / "train.csv"
    if later_path is None:
        later_path = ROOT_DIR / "data" / "raw" / "later.csv"

    # Fallback to data/ folder if raw/ files are DVC stubs
    if not train_path.exists():
        train_path = ROOT_DIR / "data" / "train.csv"
    if not later_path.exists():
        later_path = ROOT_DIR / "data" / "later.csv"

    df_train_raw = load_dataset(train_path)
    df_train = clean_training_data(df_train_raw)
    df_later = load_dataset(later_path)

    psi_results: dict[str, float] = {}

    for feature in NUMERIC_FEATURES:
        psi_val = compute_psi_for_feature(
            reference=df_train[feature],
            comparison=df_later[feature],
        )
        psi_results[feature] = psi_val
        print(f"  PSI [{feature}]: {psi_val:.6f}")

    most_drifted = max(psi_results, key=psi_results.get)
    print(f"\nMost drifted feature: {most_drifted} (PSI = {psi_results[most_drifted]:.6f})")
    return psi_results


def psi_interpretation(psi: float) -> str:
    """Human-readable interpretation of a PSI value."""
    if psi < 0.10:
        return "No significant drift"
    elif psi < 0.20:
        return "Moderate shift — monitor"
    else:
        return "Significant drift — investigate / retrain"


if __name__ == "__main__":
    print("=" * 60)
    print("PSI Drift Detection — MP033")
    print("=" * 60)
    results = compute_all_psi()
    print("\nFull PSI report:")
    print("-" * 40)
    for feat, psi_val in results.items():
        interp = psi_interpretation(psi_val)
        print(f"  {feat:<30s}: {psi_val:.6f}  — {interp}")
    most_drifted = max(results, key=results.get)
    print(f"\nConclusion: Most drifted feature = '{most_drifted}'")
