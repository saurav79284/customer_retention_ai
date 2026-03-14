# explainability/shap_explainer.py

import json
import joblib
import pandas as pd
import shap
from pathlib import Path

# =====================================================
# Paths
# =====================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # customer_retention_ai/
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.parquet"

MODEL_DIR = BASE_DIR / "models" / "artifacts"

MODEL_PATH = MODEL_DIR / "churn_lightgbm.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


# =====================================================
# Load Artifacts
# =====================================================

def load_model_and_metadata():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Model artifacts not found.")

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return model, metadata


def load_features():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Feature store not found.")

    return pd.read_parquet(FEATURES_PATH)


# =====================================================
# SHAP Logic
# =====================================================

def validate_feature_version(df: pd.DataFrame, metadata: dict):
    feature_version = df["feature_version"].iloc[0]
    expected = metadata["trained_on_feature_version"]

    if feature_version != expected:
        raise ValueError(
            f"Feature version mismatch: expected {expected}, got {feature_version}"
        )


def prepare_model_input(df: pd.DataFrame):
    X = df.drop(columns=["customer_id", "feature_version"], errors="ignore")

    if "churn" in X.columns:
        X = X.drop(columns=["churn"])

    return X


def explain_predictions(top_k: int = 5):
    """
    Returns top_k SHAP drivers per customer.
    """

    model, metadata = load_model_and_metadata()
    df = load_features()

    validate_feature_version(df, metadata)
    X = prepare_model_input(df)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, index 1 = positive class (churn)
    shap_matrix = shap_values[1] if isinstance(shap_values, list) else shap_values

    explanations = []

    for i, customer_id in enumerate(df["customer_id"]):
        values = shap_matrix[i]
        feature_names = X.columns

        shap_df = (
            pd.DataFrame({
                "feature": feature_names,
                "shap_value": values
            })
            .assign(abs_value=lambda x: x["shap_value"].abs())
            .sort_values("abs_value", ascending=False)
            .head(top_k)
            .drop(columns="abs_value")
        )

        explanations.append({
            "customer_id": customer_id,
            "top_drivers": shap_df.to_dict(orient="records")
        })

    return explanations


# =====================================================
# CLI Entry
# =====================================================

if __name__ == "__main__":
    results = explain_predictions(top_k=5)
    print(results[0])