# customer_retention_ai/decision_engine/uplift_provider.py

import json
import joblib
import pandas as pd
from pathlib import Path

# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

FEATURE_SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
PROPENSITY_MODEL_PATH = MODEL_DIR / "propensity_model.pkl"
PROPENSITY_MODEL = joblib.load(PROPENSITY_MODEL_PATH)
ACTION_MODELS = {
    "NO_ACTION": "churn_no_action.pkl",
    "DISCOUNT_10": "churn_discount_10.pkl",
    "PRIORITY_SUPPORT": "churn_priority_support.pkl",
    "LOYALTY_OFFER": "churn_loyalty_offer.pkl",
}

# =====================================================
# Load feature schema
# =====================================================

if not FEATURE_SCHEMA_PATH.exists():
    raise FileNotFoundError(
        "feature_schema.json not found. "
        "Run T-Learner training first."
    )

with open(FEATURE_SCHEMA_PATH, "r") as f:
    FEATURE_COLS = json.load(f)

# =====================================================
# Load models
# =====================================================

def load_models():
    models = {}
    for action, fname in ACTION_MODELS.items():
        path = MODEL_DIR / fname
        if path.exists():
            models[action] = joblib.load(path)
    return models


MODELS = load_models()

# =====================================================
# Uplift estimation
# =====================================================

def estimate_uplift(features: pd.DataFrame) -> dict:
    X = features[FEATURE_COLS].astype(float)

    # Predict outcome probabilities
    outcome_probs = {
        action: model.predict_proba(X)[0, 1]
        for action, model in MODELS.items()
    }

    # Predict treatment probabilities
    propensities = dict(
        zip(
            PROPENSITY_MODEL.classes_,
            PROPENSITY_MODEL.predict_proba(X)[0]
        )
    )

    base = outcome_probs["NO_ACTION"]

    uplift = {}
    for action, prob in outcome_probs.items():
        if action == "NO_ACTION":
            continue

        # Inverse Propensity Weighting
        weight = 1.0 / max(propensities.get(action, 1e-3), 1e-3)
        uplift[action] = (base - prob) * weight

    return uplift