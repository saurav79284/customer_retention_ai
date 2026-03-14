# models/predict.py

import json
import uuid
import joblib
import pandas as pd
from pathlib import Path

# =====================================================
# Paths
# =====================================================
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = MODEL_DIR / "churn_lightgbm.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.parquet"




# =====================================================
# Load Artifacts
# =====================================================

def load_model_and_metadata():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Train the model first."
        )

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return model, metadata


def load_features():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Feature store not found. Run feature engineering first."
        )

    return pd.read_parquet(FEATURES_PATH)


# =====================================================
# Inference Logic
# =====================================================

def validate_feature_version(df: pd.DataFrame, model_metadata: dict):
    feature_version = df["feature_version"].iloc[0]
    expected_version = model_metadata["trained_on_feature_version"]

    if feature_version != expected_version:
        raise ValueError(
            f"Feature version mismatch: "
            f"model expects {expected_version}, "
            f"got {feature_version}"
        )

    return feature_version


def prepare_model_input(df: pd.DataFrame):
    X = df.drop(columns=["customer_id", "feature_version"], errors="ignore")

    # Remove target if present (safety)
    if "churn" in X.columns:
        X = X.drop(columns=["churn"])

    return X


def predict_churn(save: bool = False):
    """
    Batch inference on the feature store.
    Returns churn probability per customer.
    """

    request_id = str(uuid.uuid4())

    model, metadata = load_model_and_metadata()
    df = load_features()

    feature_version = validate_feature_version(df, metadata)
    X = prepare_model_input(df)

    churn_probs = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "request_id": request_id,
        "customer_id": df["customer_id"],
        "feature_version": feature_version,
        "model_version": metadata["model_version"],
        "churn_probability": churn_probs
    })
    if save:
        results.to_parquet("data/processed/predictions.parquet", index=False)
    return results


# =====================================================
# CLI Entry
# =====================================================

if __name__ == "__main__":
    predictions = predict_churn(save=True)
    print(predictions.head())