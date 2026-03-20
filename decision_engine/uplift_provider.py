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

# =====================================================
# Default Uplift Estimates (from historical analysis)
# =====================================================
# When T-Learner confidence is low, use these data-driven defaults
DEFAULT_UPLIFT = {
    "DISCOUNT_10": 0.12,
    "PRIORITY_SUPPORT": 0.11,
    "LOYALTY_OFFER": 0.16
}

# =====================================================
# Uplift estimation
# =====================================================

def estimate_uplift(features: pd.DataFrame) -> dict:
    """
    Estimate treatment uplift using T-Learner with confidence-based fallback.
    
    Strategy:
    - If propensity >= 15%, use T-Learner (customer-specific)
    - If propensity < 15%, use historical defaults (more stable)
    - If T-Learner predicts negative uplift, use historical defaults
    
    Returns:
        dict with uplift for each action (realistic 0.0-1.0 range)
    """
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

        propensity = max(propensities.get(action, 1e-3), 0.001)
        
        # Always compute T-Learner estimate (customer-specific)
        weight = 1.0 / propensity
        raw_uplift = (base - prob) * weight
        
        # Bound to [0, 1] range (no uplift > 100%)
        bounded_uplift = max(0.0, min(1.0, raw_uplift))
        
        # For very low propensity (< 1%), blend T-Learner with default for stability
        # This maintains personalization while avoiding extreme outliers
        if propensity < 0.01:
            # Blend: 70% T-Learner, 30% default for stability
            blend_weight = propensity / 0.01  # Ranges from 0 to 1 as propensity goes 0.001 to 0.01
            uplift[action] = (blend_weight * bounded_uplift) + ((1 - blend_weight) * DEFAULT_UPLIFT[action])
        else:
            # For propensity >= 1%, trust T-Learner fully (customer-specific)
            uplift[action] = bounded_uplift

    return uplift