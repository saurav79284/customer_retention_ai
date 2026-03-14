# customer_retention_ai/training/train_t_learner.py

import json
from pathlib import Path

import pandas as pd

from customer_retention_ai.training.load_data import load_telco_data
from customer_retention_ai.training.simulate_actions import add_action_column
from customer_retention_ai.training.train_single_model import train_action_model


# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SCHEMA_PATH = MODEL_DIR / "feature_schema.json"


# =====================================================
# Main Training Logic
# =====================================================

def main():
    # ---------------------------------
    # Load feature store (WITH churn)
    # ---------------------------------
    df = load_telco_data(FEATURES_PATH)

    # ---------------------------------
    # Simulate treatments / actions
    # ---------------------------------
    df = add_action_column(df)

    # ---------------------------------
    # Define feature columns
    # ---------------------------------
    NON_FEATURE_COLUMNS = {
        "customer_id",
        "feature_version",
        "churn",
        "action",
    }

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLUMNS
    ]

    # ---------------------------------
    # SAVE FEATURE SCHEMA (CRITICAL)
    # ---------------------------------
    with open(FEATURE_SCHEMA_PATH, "w") as f:
        json.dump(feature_cols, f)

    print(f"✅ Saved feature schema at {FEATURE_SCHEMA_PATH}")

    # ---------------------------------
    # Prepare X / y
    # ---------------------------------
    X = df[feature_cols]
    y = df["churn"]

    # ---------------------------------
    # Train one model per action
    # ---------------------------------
    for action in df["action"].unique():
        mask = df["action"] == action
        n_samples = mask.sum()

        if n_samples < 100:
            print(f"⚠️ Skipping {action} (only {n_samples} samples)")
            continue

        print(f"🚀 Training model for action: {action} ({n_samples} samples)")

        train_action_model(
            X[mask],
            y[mask],
            action_name=action,
            model_dir=MODEL_DIR,
        )

    print("🎉 T-Learner training completed successfully")


# =====================================================
# CLI Entry
# =====================================================

if __name__ == "__main__":
    main()