import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# ======================================================
# Path setup (keep as-is)
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ======================================================
# Imports
# ======================================================

from customer_retention_ai.models.predict import predict_churn, load_features
from customer_retention_ai.explainability.shap_explainer import explain_predictions
from customer_retention_ai.decision_engine.policy import choose_best_action
from customer_retention_ai.decision_engine.uplift_provider import estimate_uplift
from customer_retention_ai.decision_engine.actions import get_available_actions

# ======================================================
# Page config
# ======================================================

st.set_page_config(
    page_title="Customer Retention Decision Console",
    layout="wide"
)

st.title("🎯 Customer Retention Decision Console")

# ======================================================
# Load predictions
# ======================================================

@st.cache_data
def load_predictions():
    return predict_churn()

preds = load_predictions()

# ======================================================
# Customer selector
# ======================================================

st.sidebar.header("Customer Selection")

customer_id = st.sidebar.selectbox(
    "Select Customer ID",
    preds["customer_id"].unique()
)

row = preds[preds["customer_id"] == customer_id].iloc[0]

# ======================================================
# Churn Risk Panel
# ======================================================

st.subheader("📉 Churn Risk")

risk = float(row["churn_probability"])

st.metric(
    label="Churn Probability",
    value=f"{risk:.2%}"
)

if risk >= 0.6:
    st.error("⚠️ High Risk Customer")
elif risk >= 0.3:
    st.warning("⚠️ Medium Risk Customer")
else:
    st.success("✅ Low Risk Customer")

# ======================================================
# Explainability Panel
# ======================================================

st.subheader("🧠 Why is this customer at risk?")

explanations = explain_predictions(top_k=5)
drivers = [
    d for d in explanations
    if d["customer_id"] == customer_id
][0]["top_drivers"]

st.table(pd.DataFrame(drivers))

# ======================================================
# Decision Engine
# ======================================================

st.subheader("🤖 Recommended Action")

# Build feature vector for uplift
customer_features = (
    load_features()
    .query("customer_id == @customer_id")
    .drop(columns=["customer_id", "feature_version", "churn"], errors="ignore")
)

uplift_estimates = estimate_uplift(customer_features)

decision = choose_best_action(
    customer_id=customer_id,
    churn_probability=risk,
    customer_value=1200,  # placeholder value
    uplift_estimates=uplift_estimates
)

st.markdown(f"""
### ✅ Final Recommendation
**Action:** `{decision['recommended_action']}`  
**Expected Value (₹):** `{decision['expected_value']:.2f}`
""")

# ======================================================
# 🧠 WHY THIS ACTION (PHASE 1 ADDITION)
# ======================================================

st.subheader("🧠 Why this action?")

if decision["recommended_action"] == "NO_ACTION":
    st.info(
        "No intervention is recommended because all available actions "
        "have negative expected value after considering cost."
    )
else:
    st.success(
        f"**{decision['recommended_action']}** was selected because it "
        f"provides the highest positive expected value after accounting "
        f"for uplift and action cost."
    )

# ======================================================
# 🧮 EXPECTED VALUE BREAKDOWN (PHASE 1 ADDITION)
# ======================================================

st.subheader("🧮 Expected Value Breakdown")

ev_df = pd.DataFrame(decision["all_action_evaluations"])
ev_df = ev_df.sort_values("expected_value", ascending=False)

st.dataframe(ev_df, use_container_width=True)

# ======================================================
# 📈 UPLIFT VS COST (PHASE 1 ADDITION)
# ======================================================

st.subheader("📈 Uplift vs Cost")

rows = []
for action in get_available_actions():
    uplift = uplift_estimates.get(action.action_id, 0.0)
    ev = next(
        x["expected_value"]
        for x in decision["all_action_evaluations"]
        if x["action_id"] == action.action_id
    )

    rows.append({
        "action": action.action_id,
        "uplift": round(uplift, 4),
        "cost": action.cost,
        "expected_value": ev
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ======================================================
# Human-in-the-loop
# ======================================================

st.subheader("🧑‍💼 Human Decision")

final_action = st.selectbox(
    "Approve or Override",
    options=[
        decision["recommended_action"],
        "NO_ACTION",
        "DISCOUNT_10",
        "PRIORITY_SUPPORT",
        "LOYALTY_OFFER"
    ]
)

comment = st.text_area("Decision Notes (optional)")

if st.button("✅ Confirm Decision"):
    st.success("Decision recorded successfully.")
    st.json({
        "customer_id": customer_id,
        "model_action": decision["recommended_action"],
        "final_action": final_action,
        "expected_value": decision["expected_value"],
        "notes": comment
    })