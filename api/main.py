from fastapi import FastAPI
import uuid

from api.schemas import (
    PredictRequest, PredictResponse, DecisionResponse,
    SimulateRequest, SimulateResponse, SimulateMetricsResponse,
    ActionDistributionResponse
)
from models.predict import predict_churn
from explainability.shap_explainer import explain_predictions
from decision_engine.policy import choose_best_action
from decision_engine.ab_test_simulator import ABTestSimulator

app = FastAPI(
    title="Customer Retention Decision Intelligence API",
    version="1.0.0"
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    preds = predict_churn()
    row = preds.iloc[0]

    return PredictResponse(
        request_id=row["request_id"],
        customer_id=row["customer_id"],
        churn_probability=float(row["churn_probability"]),
        model_version=row["model_version"],
        feature_version=row["feature_version"],
    )


@app.post("/decision", response_model=DecisionResponse)
def decision(req: PredictRequest):
    preds = predict_churn()
    row = preds.iloc[0]

    explanations = explain_predictions(top_k=5)
    drivers = explanations[0]["top_drivers"]
    
    # Get customer-specific uplift estimates
    # Include necessary imports at the top of the file
    from customer_retention_ai.models.predict import load_features
    from customer_retention_ai.decision_engine.uplift_provider import estimate_uplift
    
    customer_features = (
        load_features()
        .query("customer_id == @row['customer_id']")
        .drop(columns=["customer_id", "feature_version", "churn"], errors="ignore")
    )
    
    # Get dynamic uplift for this specific customer
    # Falls back to historical defaults if T-Learner confidence is low
    uplift_estimates = estimate_uplift(customer_features)

    decision = choose_best_action(
        customer_id=row["customer_id"],
        churn_probability=row["churn_probability"],
        customer_value=1200,
        uplift_estimates=uplift_estimates
    )

    return DecisionResponse(
        request_id=row["request_id"],
        customer_id=row["customer_id"],
        churn_probability=float(row["churn_probability"]),
        model_version=row["model_version"],
        feature_version=row["feature_version"],
        top_drivers=drivers,
        recommended_action=decision["recommended_action"],
        expected_value=decision["expected_value"]
    )


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    """
    Run A/B test simulation on retention policy.
    
    Args:
        n_customers: Number of customers to simulate (default: 1000)
        customer_value: Customer lifetime value (default: 1200)
        uplift_*: Uplift rates for each action
    
    Returns:
        SimulateResponse with metrics, action distribution, and summary
    """

    simulation_id = str(uuid.uuid4())

    # Initialize simulator
    simulator = ABTestSimulator(
        customer_value=req.customer_value,
        random_state=42
    )

    # Define uplift estimates
    uplift_estimates = {
        "DISCOUNT_10": req.uplift_discount_10,
        "PRIORITY_SUPPORT": req.uplift_priority_support,
        "LOYALTY_OFFER": req.uplift_loyalty_offer
    }

    # Run simulation
    results = simulator.run_simulation(
        n_customers=req.n_customers,
        uplift_estimates=uplift_estimates
    )

    metrics = results["metrics"]
    action_dist = results["action_distribution"]

    # Normalize action distribution (fill missing actions)
    normalized_dist = {
        "NO_ACTION": action_dist.get("NO_ACTION", 0),
        "DISCOUNT_10": action_dist.get("DISCOUNT_10", 0),
        "PRIORITY_SUPPORT": action_dist.get("PRIORITY_SUPPORT", 0),
        "LOYALTY_OFFER": action_dist.get("LOYALTY_OFFER", 0)
    }

    # Campaign summary
    campaign_summary = {
        "sample_size": metrics["sample_size"],
        "total_ai_revenue": metrics["ai_total_revenue"],
        "total_ai_cost": metrics["ai_total_cost"],
        "net_ai_revenue": metrics["ai_net_revenue"],
        "expected_roi": metrics["roi_percentage"],
        "revenue_lift": metrics["revenue_lift_percentage"],
        "significance": metrics["statistically_significant"]
    }

    return SimulateResponse(
        simulation_id=simulation_id,
        metrics=SimulateMetricsResponse(**metrics),
        action_distribution=ActionDistributionResponse(**normalized_dist),
        campaign_summary=campaign_summary
    )