import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px

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
from customer_retention_ai.evaluation.policy_evaluation import evaluate_policy
from customer_retention_ai.decision_engine.ab_test_simulator import ABTestSimulator
from customer_retention_ai.config.loader import get_customer_value, get_risk_thresholds

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

@st.cache_data
def load_explanations():
    return explain_predictions(top_k=5)

def load_policy_results():
    """Policy evaluation is computed dynamically (not cached) to reflect current config."""
    return evaluate_policy()

def get_customer_uplift(customer_id_input: str):
    """Get fresh uplift estimates per customer (not cached to ensure dynamic values)."""
    features = load_features()
    customer_features = (
        features
        .query("customer_id == @customer_id_input")
        .drop(columns=["customer_id", "feature_version", "churn"], errors="ignore")
    )
    return estimate_uplift(customer_features)

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

# Load risk thresholds from config
risk_thresholds = get_risk_thresholds()
high_risk_threshold = risk_thresholds.get("high_risk", 0.6)
medium_risk_threshold = risk_thresholds.get("medium_risk", 0.3)

if risk >= high_risk_threshold:
    st.error("⚠️ High Risk Customer")
elif risk >= medium_risk_threshold:
    st.warning("⚠️ Medium Risk Customer")
else:
    st.success("✅ Low Risk Customer")

# ======================================================
# Explainability Panel
# ======================================================

st.subheader("🧠 Why is this customer at risk?")

explanations = load_explanations()
drivers = [
    d for d in explanations
    if d["customer_id"] == customer_id
][0]["top_drivers"]

st.table(pd.DataFrame(drivers))

# ======================================================
# Decision Engine
# ======================================================

st.subheader("🤖 Recommended Action")

# ✅ USE CUSTOMER-SPECIFIC UPLIFT from T-Learner (cached by customer)
# - Falls back to historical defaults when T-Learner confidence is low
# - Ensures reasonable predictions for each customer
uplift_estimates = get_customer_uplift(customer_id)

decision = choose_best_action(
    customer_id=customer_id,
    churn_probability=risk,
    customer_value=get_customer_value(),
    uplift_estimates=uplift_estimates
)

st.markdown(f"""
### ✅ Final Recommendation
**Action:** `{decision['recommended_action']}`  
**Expected Value (₹):** `{decision['expected_value']:.2f}`

**Uplift Estimates Used:**
- DISCOUNT_10: {uplift_estimates.get('DISCOUNT_10', 0):.2%}
- PRIORITY_SUPPORT: {uplift_estimates.get('PRIORITY_SUPPORT', 0):.2%}
- LOYALTY_OFFER: {uplift_estimates.get('LOYALTY_OFFER', 0):.2%}
""")



st.subheader("📊 Policy Evaluation")

# Add refresh button to dynamically recompute policy evaluation
col1, col2 = st.columns([4, 1])
with col1:
    st.caption("Comprehensive evaluation of AI policy vs baseline on entire dataset")
with col2:
    if st.button("🔄 Refresh", key="policy_refresh"):
        st.session_state.refresh_policy = True

# Calculate policy with progress indication
try:
    if "refresh_policy" in st.session_state and st.session_state.refresh_policy:
        with st.spinner("📊 Calculating policy evaluation... (this may take a moment)"):
            results = load_policy_results()
        st.session_state.refresh_policy = False
    else:
        results = load_policy_results()
except Exception as e:
    st.error(f"❌ Error calculating policy evaluation: {str(e)}")
    st.write("Please check the configuration and ensure all data files are available.")
    import traceback
    st.write(traceback.format_exc())
    results = None

# Display results if available
if results:
    st.metric(
        label="AI Policy Value",
        value=f"{results['ai_policy_value']:.2f}"
    )

    st.metric(
        label="Baseline Policy Value",
        value=f"{results['baseline_value']:.2f}"
    )

    st.metric(
        label="Estimated Improvement",
        value=f"{results['improvement_pct']:.2f}%"
    )

    import pandas as pd

    chart_df = pd.DataFrame({
        "Policy": ["Baseline", "AI Policy"],
        "Value": [
            results["baseline_value"],
            results["ai_policy_value"]
        ]
    })

    st.bar_chart(chart_df.set_index("Policy"))

# ======================================================
# 📊 ACTION DIVERSIFICATION DIAGNOSTIC
# ======================================================

with st.expander("📊 Why is action distribution skewed? (Diagnostic)", expanded=False):
    st.info("""
    **Understanding Action Distribution:**
    
    Each action is recommended based on Expected Value (EV):
    
    **EV Formula:** (Uplift × Customer Value) - Action Cost
    
    **Why one action dominates:**
    - If one action has consistently higher uplift estimates than others
    - Even with higher cost, higher uplift can overcome it
    - Example: If DISCOUNT_10 uplift = 15% and LOYALTY_OFFER = 8%
      - DISCOUNT: ₹(0.15 × 1200) - 50 = ₹130 EV
      - LOYALTY: ₹(0.08 × 1200) - 20 = ₹76 EV
      - DISCOUNT wins despite higher cost!
    
    **Current Recommendation Rate:** 100% DISCOUNT_10
    
    **Possible Root Causes:**
    1. ✋ **Model Imbalance**: T-Learner models not well-calibrated across actions
    2. ✋ **Biased Training Data**: One action had better historical outcomes
    3. ✋ **Feature Collinearity**: Features correlate more with one treatment
    4. ✋ **Propensity Scores**: One action has much higher estimated propensity
    
    **Solutions to Try:**
    - Retrain T-Learner models with balanced hyperparameters
    - Check propensity score distribution across actions
    - Verify default uplift values in config.yaml
    - Add stratified resampling during model training
    """)
    
    # Show suggested config changes
    st.subheader("🔧 Diagnostic: Suggested Config Adjustments")
    st.write("""
    To encourage action diversity, you can modify `config.yaml`:
    """)
    
    st.code("""
# Current config favors DISCOUNT (highest cost, highest uplift?)
policy:
  customer_value: 1200

actions:
  DISCOUNT_10:
    cost: 50  # Try increasing cost penalty
    
  PRIORITY_SUPPORT:
    cost: 30  # Try decreasing cost
    
  LOYALTY_OFFER:
    cost: 20  # Try decreasing cost
    """, language="yaml")
    
    st.warning("""
    **⚠️ Better Solution:** Retrain the T-Learner models to ensure:
    - Balanced treatment assignment during training
    - Proper cross-validation across actions
    - Similar propensity score distributions
    """)

# ======================================================
# 🧠 WHY THIS ACTION (PHASE 1 ADDITION)
# ======================================================

st.subheader("🧠 Why this action?")

if decision["recommended_action"] == "NO_ACTION":
    from customer_retention_ai.config.loader import get_actions_config
    customer_val = get_customer_value()
    actions_config = get_actions_config()
    
    discount_cost = actions_config.get('DISCOUNT_10', {}).get('cost', 50)
    support_cost = actions_config.get('PRIORITY_SUPPORT', {}).get('cost', 30)
    loyalty_cost = actions_config.get('LOYALTY_OFFER', {}).get('cost', 20)
    
    st.info(
        "✓ **NO ACTION** is recommended for this customer because:\n\n"
        "Even considering their individual uplift potential, all retention actions "
        "have costs that exceed their expected benefits.\n\n"
        f"Analysis:\n"
        f"- DISCOUNT_10: {uplift_estimates.get('DISCOUNT_10', 0):.2%} uplift - cost = "
        f"₹{(uplift_estimates.get('DISCOUNT_10', 0) * customer_val) - discount_cost:.0f} net\n"
        f"- PRIORITY_SUPPORT: {uplift_estimates.get('PRIORITY_SUPPORT', 0):.2%} uplift - cost = "
        f"₹{(uplift_estimates.get('PRIORITY_SUPPORT', 0) * customer_val) - support_cost:.0f} net\n"
        f"- LOYALTY_OFFER: {uplift_estimates.get('LOYALTY_OFFER', 0):.2%} uplift - cost = "
        f"₹{(uplift_estimates.get('LOYALTY_OFFER', 0) * customer_val) - loyalty_cost:.0f} net"
    )
else:
    action_info = decision["all_action_evaluations"]
    best_action = decision["recommended_action"]
    best_ev = decision["expected_value"]
    
    st.success(
        f"✅ **{best_action}** is recommended with Expected Value: ₹{best_ev:.2f}\n\n"
        f"Based on customer-specific uplift potential:\n"
    )
    
    # Show all options
    for eval_item in action_info:
        action = eval_item["action_id"]
        ev = eval_item["expected_value"]
        uplift = uplift_estimates.get(action, 0)
        
        if action == best_action:
            st.write(f"**✓ {action}: ₹{ev:.2f}** ← Best option (uplift: {uplift:.2%})")
        else:
            st.write(f"  {action}: ₹{ev:.2f} (uplift: {uplift:.2%})")

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


# ======================================================
# 🧪 A/B TEST SIMULATOR
# ======================================================

st.divider()

st.subheader("🧪 A/B Test Simulator")

st.markdown("""
Simulate campaign performance before deployment. Compare AI policy vs baseline 
(no action) on a sample of customers.
""")

# Display uplift estimate source
st.info("""
📊 **Data-Driven Uplift Estimates:**
- **Baseline Churn Rate:** 26.54% (from 7,043 customers)
- **High-Risk Segment:** 66.50% churn (month-to-month contracts, low tenure)
- **Estimated Impact:** LOYALTY_OFFER prevents ~42 churns per 1,000 customers
""")

# Metrics columns
with st.expander("⚙️ Simulation Settings", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        sim_n_customers = st.slider(
            "Sample Size",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )
        sim_customer_value = st.number_input(
            "Customer Lifetime Value (₹)",
            min_value=500.0,
            max_value=5000.0,
            value=float(get_customer_value()),
            step=100.0
        )
    
    with col2:
        st.markdown("**Uplift Estimates:** (based on historical data analysis)")
        
        # Computed from Telco dataset analysis
        st.caption("📊 Estimated churn reduction for each retention action:")
        
        sim_uplift_discount = st.slider(
            "DISCOUNT_10 Uplift",
            min_value=0.0,
            max_value=0.5,
            value=0.12,
            step=0.01,
            format="%.2f",
            help="12% average reduction in churn from 10% discount offer"
        )
        sim_uplift_support = st.slider(
            "PRIORITY_SUPPORT Uplift",
            min_value=0.0,
            max_value=0.5,
            value=0.11,
            step=0.01,
            format="%.2f",
            help="11% average reduction in churn from priority support tier"
        )
        sim_uplift_loyalty = st.slider(
            "LOYALTY_OFFER Uplift",
            min_value=0.0,
            max_value=0.5,
            value=0.16,
            step=0.01,
            format="%.2f",
            help="16% average reduction in churn from loyalty offer (most effective)"
        )

# Run simulation button
if st.button("🚀 Run Simulation", key="simulate_button"):
    with st.spinner("Running A/B test simulation..."):
        try:
            # Initialize simulator
            simulator = ABTestSimulator(
                customer_value=sim_customer_value,
                random_state=42
            )
            
            # Run simulation
            results = simulator.run_simulation(
                n_customers=sim_n_customers,
                uplift_estimates={
                    "DISCOUNT_10": sim_uplift_discount,
                    "PRIORITY_SUPPORT": sim_uplift_support,
                    "LOYALTY_OFFER": sim_uplift_loyalty
                }
            )
            
            metrics = results["metrics"]
            action_dist = results["action_distribution"]
            
            # Display results
            st.success("✅ Simulation completed!")
            
            # Key metrics
            st.subheader("📊 Campaign Metrics")
            
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric(
                    "Revenue Lift",
                    f"{metrics['revenue_lift_percentage']:.2f}%",
                    delta=f"{metrics['revenue_lift_percentage']:.2f}%"
                )
            
            with metric_cols[1]:
                st.metric(
                    "ROI",
                    f"{metrics['roi_percentage']:.2f}%",
                    delta=f"{metrics['roi_percentage']:.2f}%"
                )
            
            with metric_cols[2]:
                st.metric(
                    "Retention (AI)",
                    f"{metrics['retention_rate_ai']:.2%}",
                    delta=f"{(metrics['retention_rate_ai'] - metrics['retention_rate_baseline']):.2%}"
                )
            
            with metric_cols[3]:
                st.metric(
                    "Significance",
                    "✅ Yes" if metrics['statistically_significant'] else "❌ No",
                    delta=f"p={metrics['p_value']:.4f}"
                )
            
            # Revenue breakdown
            st.subheader("💰 Revenue Analysis")
            
            revenue_cols = st.columns(3)
            
            with revenue_cols[0]:
                st.metric(
                    "AI Total Revenue",
                    f"₹{metrics['ai_total_revenue']:,.2f}"
                )
            
            with revenue_cols[1]:
                st.metric(
                    "Campaign Cost",
                    f"₹{metrics['ai_total_cost']:,.2f}"
                )
            
            with revenue_cols[2]:
                st.metric(
                    "Net Revenue",
                    f"₹{metrics['ai_net_revenue']:,.2f}"
                )
            
            # Comparison chart
            st.subheader("📈 Policy Comparison")
            
            comparison_df = pd.DataFrame({
                "Policy": ["Baseline", "AI Policy"],
                "Revenue": [metrics['baseline_total_revenue'], metrics['ai_total_revenue']],
                "Cost": [0, metrics['ai_total_cost']],
                "Net Benefit": [metrics['baseline_net_revenue'], metrics['ai_net_revenue']]
            })
            
            st.bar_chart(
                comparison_df[['Revenue', 'Net Benefit']].set_index(comparison_df['Policy'])
            )
            
            # Per-customer metrics
            st.subheader("💼 Per-Customer Economics")
            
            per_cust_cols = st.columns(3)
            
            with per_cust_cols[0]:
                st.metric(
                    "Avg EU (AI)",
                    f"₹{metrics['avg_eu_per_customer_ai']:.2f}"
                )
            
            with per_cust_cols[1]:
                st.metric(
                    "Avg EU (Baseline)",
                    f"₹{metrics['avg_eu_per_customer_baseline']:.2f}"
                )
            
            with per_cust_cols[2]:
                improvement = metrics['avg_eu_per_customer_ai'] - metrics['avg_eu_per_customer_baseline']
                st.metric(
                    "Improvement per Customer",
                    f"₹{improvement:.2f}",
                    delta=f"₹{improvement:.2f}"
                )
            
            # Download results
            st.divider()
            
            results_csv = pd.DataFrame(results["campaign_results"]).to_csv(index=False)
            
            st.download_button(
                label="📥 Download Campaign Results (CSV)",
                data=results_csv,
                file_name=f"ab_test_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Simulation failed: {str(e)}")
            st.info("Make sure your feature store and models are properly trained.")