"""
monitoring.py — Master monitoring pipeline for MP033.

Owner: Kabir Nagdev (monitoring track)

Orchestrates:
  1. PSI drift detection       → drift.py
  2. Group-wise MAE evaluation → evaluation.py
  3. Temporal v1/v2 retraining → retraining.py
  4. Holdout predictions        → holdout.py
  5. findings.yaml update       → monitoring section only

Run this script to execute the full monitoring pipeline end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift import compute_all_psi, psi_interpretation  # noqa: E402
from evaluation import run_group_evaluation  # noqa: E402
from retraining import run_temporal_retraining  # noqa: E402
from holdout import generate_holdout_predictions  # noqa: E402

FINDINGS_PATH = ROOT_DIR / "findings.yaml"


# ---------------------------------------------------------------------------
# findings.yaml helpers — ONLY monitoring section is modified
# ---------------------------------------------------------------------------

def _load_findings() -> dict:
    """Load existing findings.yaml preserving all sections."""
    if not FINDINGS_PATH.exists():
        raise FileNotFoundError(f"findings.yaml not found at {FINDINGS_PATH}")
    with open(FINDINGS_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _save_findings(data: dict) -> None:
    """Write findings.yaml back to disk."""
    with open(FINDINGS_PATH, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _update_monitoring_section(
    findings: dict,
    psi_results: dict[str, float],
    mae_by_block: dict[str, float],
    worst_block: str,
    mae_v1: float,
    mae_v2: float,
    decision: str,
) -> dict:
    """
    Populate the monitoring section of findings.yaml.
    Does NOT touch data_issues or tracking sections.
    """
    most_drifted = max(psi_results, key=psi_results.get)

    findings["monitoring"] = {
        "psi": {
            "students_on_campus": psi_results["students_on_campus"],
            "menu_price_rs": psi_results["menu_price_rs"],
            "temp_c": psi_results["temp_c"],
            "exam_days_left": psi_results["exam_days_left"],
            "special_dishes": psi_results["special_dishes"],
            "queue_minutes_yesterday": psi_results["queue_minutes_yesterday"],
        },
        "most_drifted_feature": most_drifted,
        "mae_by_group": {
            "A-Block": mae_by_block.get("A-Block"),
            "B-Block": mae_by_block.get("B-Block"),
            "Hostel": mae_by_block.get("Hostel"),
        },
        "worst_served_group": worst_block,
        "mae_v1_on_check": mae_v1,
        "mae_v2_on_check": mae_v2,
        "decision": decision,
    }
    return findings


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_monitoring_pipeline() -> None:
    """Execute the full monitoring pipeline in sequence."""

    print("\n" + "=" * 60)
    print("STEP 1 / 4 — PSI Drift Detection")
    print("=" * 60)
    psi_results = compute_all_psi()

    print("\n" + "=" * 60)
    print("STEP 2 / 4 — Group-wise MAE Evaluation (v1 on later.csv)")
    print("=" * 60)
    v1_model, mae_by_block, worst_block = run_group_evaluation()

    print("\n" + "=" * 60)
    print("STEP 3 / 4 — Temporal Retraining & Promotion (v1 vs v2)")
    print("=" * 60)
    retrain_result = run_temporal_retraining(v1_model=v1_model)

    print("\n" + "=" * 60)
    print("STEP 4 / 4 — Holdout Predictions")
    print("=" * 60)
    generate_holdout_predictions()

    # ---- Update findings.yaml (monitoring section only) ----
    print("\n" + "=" * 60)
    print("Updating findings.yaml — monitoring section only")
    print("=" * 60)
    findings = _load_findings()

    # Guard: do not overwrite non-monitoring sections
    preserved_keys = {"data_issues", "tracking"}
    for key in preserved_keys:
        if key in findings:
            print(f"  Preserving section: {key}")

    findings = _update_monitoring_section(
        findings=findings,
        psi_results=psi_results,
        mae_by_block=mae_by_block,
        worst_block=worst_block,
        mae_v1=retrain_result["mae_v1"],
        mae_v2=retrain_result["mae_v2"],
        decision=retrain_result["decision"],
    )
    _save_findings(findings)
    print("  findings.yaml updated at " + str(FINDINGS_PATH))

    # ---- Final summary ----
    print("\n" + "=" * 60)
    print("MONITORING PIPELINE COMPLETE — MP033 SUMMARY")
    print("=" * 60)

    most_drifted = max(psi_results, key=psi_results.get)
    print("\nPSI Results:")
    for feat, psi_val in psi_results.items():
        flag = " <- MOST DRIFTED" if feat == most_drifted else ""
        print(f"  {feat:<30s}: {psi_val:.6f}  {psi_interpretation(psi_val)}{flag}")

    print("\nMAE by Block (v1 on later.csv):")
    for block, mae in mae_by_block.items():
        flag = " <- WORST" if block == worst_block else ""
        print(f"  {block:<10s}: {mae:.4f}{flag}")

    print(f"\nv1 MAE on check set : {retrain_result['mae_v1']:.4f}")
    print(f"v2 MAE on check set : {retrain_result['mae_v2']:.4f}")
    print(f"Improvement         : {retrain_result['improvement_pct']:.2f}%")
    print(f"Decision            : {retrain_result['decision']}")
    print(f"Champion            : {retrain_result['champion']}")


if __name__ == "__main__":
    run_monitoring_pipeline()
