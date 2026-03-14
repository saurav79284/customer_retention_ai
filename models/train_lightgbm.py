# models/train_lightgbm.py

import json
import joblib
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# =====================================================
# Paths & Config
# =====================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # customer_retention_ai/
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
MODEL_DIR = Path("models/artifacts")
MODEL_PATH = MODEL_DIR / "churn_lightgbm.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

RANDOM_STATE = 42


# =====================================================
# Training Logic
# =====================================================

def load_features():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Feature store not found. Run feature engineering first.")

    df = pd.read_parquet(FEATURES_PATH)
    return df


def split_features_target(df: pd.DataFrame):
    if "churn" not in df.columns:
        raise ValueError("Training requires churn label in feature store.")

    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    feature_version = df["feature_version"].iloc[0]

    # Drop feature_version from model input (metadata only)
    X = X.drop(columns=["feature_version"])

    return X, y, feature_version


def train_model(X, y):
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)

    return model, auc


def save_artifacts(model, feature_version, auc):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": "churn_lightgbm",
        "model_version": "v1.0.0",
        "trained_on_feature_version": feature_version,
        "validation_auc": round(float(auc), 4)
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# =====================================================
# CLI Entry
# =====================================================

if __name__ == "__main__":
    df = load_features()
    X, y, feature_version = split_features_target(df)

    model, auc = train_model(X, y)
    metadata = save_artifacts(model, feature_version, auc)

    print("Model training completed successfully")
    print(json.dumps(metadata, indent=2))