import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from pathlib import Path

# ===============================
# Paths
# ===============================

REFERENCE_FEATURES_PATH = Path("data/processed/features.parquet")
CURRENT_FEATURES_PATH = Path("data/processed/features.parquet")

REFERENCE_PRED_PATH = Path("data/processed/predictions.parquet")
CURRENT_PRED_PATH = Path("data/processed/predictions.parquet")

REPORT_PATH = Path("monitoring/drift_report.csv")

# ===============================
# Thresholds (Business Tunable)
# ===============================

KS_ALPHA = 0.05
FEATURE_DRIFT_ALERT_RATIO = 0.3  # 30% of features drifted


# ===============================
# Loaders
# ===============================

def load_features():
    ref = pd.read_parquet(REFERENCE_FEATURES_PATH)
    cur = pd.read_parquet(CURRENT_FEATURES_PATH)

    drop_cols = ["customer_id", "feature_version", "churn"]
    ref = ref.drop(columns=drop_cols, errors="ignore")
    cur = cur.drop(columns=drop_cols, errors="ignore")

    return ref, cur


def load_predictions():
    ref = pd.read_parquet(REFERENCE_PRED_PATH)
    cur = pd.read_parquet(CURRENT_PRED_PATH)

    return ref["churn_probability"], cur["churn_probability"]


# ===============================
# Drift Computations
# ===============================

def compute_feature_drift(ref: pd.DataFrame, cur: pd.DataFrame):
    results = []

    for col in ref.columns:
        if ref[col].dtype.kind in "if":
            stat, p = ks_2samp(ref[col], cur[col])
            drifted = p < KS_ALPHA

            results.append({
                "type": "feature",
                "name": col,
                "ks_stat": round(stat, 4),
                "p_value": round(p, 6),
                "drift_detected": drifted
            })

    return results


def compute_prediction_drift(ref_preds, cur_preds):
    stat, p = ks_2samp(ref_preds, cur_preds)

    return {
        "type": "prediction",
        "name": "churn_probability",
        "ks_stat": round(stat, 4),
        "p_value": round(p, 6),
        "drift_detected": p < KS_ALPHA
    }


# ===============================
# Alert Logic
# ===============================

def evaluate_alerts(drift_rows):
    feature_drifts = [
        r for r in drift_rows
        if r["type"] == "feature" and r["drift_detected"]
    ]

    total_features = len([
        r for r in drift_rows if r["type"] == "feature"
    ])

    feature_drift_ratio = (
        len(feature_drifts) / total_features
        if total_features > 0 else 0
    )

    prediction_drift = any(
        r["type"] == "prediction" and r["drift_detected"]
        for r in drift_rows
    )

    alert = (
        feature_drift_ratio >= FEATURE_DRIFT_ALERT_RATIO
        or prediction_drift
    )

    return {
        "feature_drift_ratio": round(feature_drift_ratio, 2),
        "prediction_drift": prediction_drift,
        "alert_triggered": alert
    }


# ===============================
# Main Runner
# ===============================

def run_monitoring():
    ref_features, cur_features = load_features()
    ref_preds, cur_preds = load_predictions()

    drift_rows = []
    drift_rows.extend(compute_feature_drift(ref_features, cur_features))
    drift_rows.append(compute_prediction_drift(ref_preds, cur_preds))

    alert_summary = evaluate_alerts(drift_rows)

    report = pd.DataFrame(drift_rows)
    report = pd.concat(
        [report, pd.DataFrame([{
            "type": "SYSTEM",
            "name": "OVERALL",
            **alert_summary
        }])],
        ignore_index=True
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print("Drift report saved to:", REPORT_PATH.resolve())
    print("\nALERT SUMMARY:")
    print(alert_summary)


if __name__ == "__main__":
    run_monitoring()