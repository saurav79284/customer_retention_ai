# diagnosis/debug_tlearner.py
"""Debug T-Learner model predictions"""

import pandas as pd
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from customer_retention_ai.models.predict import load_features
from customer_retention_ai.decision_engine.uplift_provider import FEATURE_COLS, MODELS, PROPENSITY_MODEL

print("=" * 70)
print("🔬 T-LEARNER DEBUG - Investigating Negative Uplift Issue")
print("=" * 70)
print()

# Load data
features_df = load_features()
print(f"✓ Loaded {len(features_df):,} customers")
print(f"✓ Feature columns loaded: {len(FEATURE_COLS)}")
print(f"✓ Models loaded: {list(MODELS.keys())}")
print(f"✓ Propensity model loaded: {PROPENSITY_MODEL.__class__.__name__}")
print()

# Test on first customer
customer_idx = 0
customer_features = features_df.drop(columns=["customer_id", "feature_version", "churn"], errors="ignore").iloc[[customer_idx]]
X = customer_features[FEATURE_COLS].astype(float)

print(f"Testing on customer {customer_idx}:")
print(f"  Features shape: {X.shape}")
print(f"  Feature ranges: min={X.values.min():.4f}, max={X.values.max():.4f}, mean={X.values.mean():.4f}")
print()

# Check each model's predictions
print("📊 Model Predictions (Churn Probability):")
outcome_probs = {}
for action, model in MODELS.items():
    try:
        prob = model.predict_proba(X)[0, 1]
        outcome_probs[action] = prob
        print(f"  {action:20s}: {prob:.4f} ({prob:.2%})")
    except Exception as e:
        print(f"  {action:20s}: ERROR - {e}")
print()

# Check propensity model
print("📊 Propensity Model Predictions:")
try:
    propensities_raw = PROPENSITY_MODEL.predict_proba(X)[0]
    propensities = dict(zip(PROPENSITY_MODEL.classes_, propensities_raw))
    
    for action, prob in propensities.items():
        print(f"  {action:20s}: {prob:.4f} ({prob:.2%})")
except Exception as e:
    print(f"  ERROR: {e}")
print()

# Compute uplift step-by-step
print("🧮 Uplift Calculation Step-by-Step:")
base = outcome_probs.get("NO_ACTION", 0.5)
print(f"  Base churn (NO_ACTION): {base:.4f} ({base:.2%})")
print()

for action in ["DISCOUNT_10", "PRIORITY_SUPPORT", "LOYALTY_OFFER"]:
    prob = outcome_probs.get(action, 0.5)
    propensity = propensities.get(action, 0.01)
    
    print(f"  {action}:")
    print(f"    - Action churn prob: {prob:.4f} ({prob:.2%})")
    print(f"    - Propensity:        {propensity:.4f} ({propensity:.2%})")
    
    raw_difference = base - prob
    print(f"    - (Base - Action):   {raw_difference:.4f}")
    
    if propensity > 0:
        weight = 1.0 / max(propensity, 0.01)
        print(f"    - IPW weight:        {weight:.4f}")
        
        raw_uplift = raw_difference * weight
        print(f"    - Raw uplift:        {raw_uplift:.4f} ({raw_uplift:.2%})")
    print()

# =====================================================
# Diagnosis
# =====================================================

print("=" * 70)
print("🔍 DIAGNOSIS")
print("=" * 70)
print()

issues = []

# Check for model quality
if all(abs(outcome_probs.get(a, 0.5) - base) < 0.01 for a in MODELS.keys()):
    issues.append("⚠️  Models predict nearly identical churn rates for all actions")
    print("   (This means treatments have no differentiated effect)")

# Check propensity scores
if any(propensities.get(a, 0.5) < 0.05 for a in ["DISCOUNT_10", "PRIORITY_SUPPORT", "LOYALTY_OFFER"]):
    issues.append("⚠️  Some propensity scores are very low (<5%)")
    print("   (This causes extreme IPW weights)")

if not issues:
    print("✅ Models appear well-trained")
else:
    print(f"\n❌ Found {len(issues)} issues:")
    for issue in issues:
        print(f"   {issue}")

print()
print("=" * 70)
print("💡 RECOMMENDATION")
print("=" * 70)
if not issues:
    print("""
✓ T-Learner models are GOOD - use them for customer-specific predictions!
  Recommendation: Re-enable estimate_uplift() in Streamlit""")
else:
    print("""
❌ T-Learner models have issues. Options:
  1. Use default aggregate uplift (12%, 11%, 16%) from historical data
  2. Retrain T-Learner with better feature engineering
  3. Use hybrid: T-Learner for high-confidence predictions + defaults for low-confidence
""")
