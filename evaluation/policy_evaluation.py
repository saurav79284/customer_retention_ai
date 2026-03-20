import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from customer_retention_ai.training.simulate_actions import add_action_column
from customer_retention_ai.decision_engine.policy import choose_best_action
from customer_retention_ai.decision_engine.uplift_provider import estimate_uplift
from customer_retention_ai.decision_engine.ab_test_simulator import OutcomeSimulator
from customer_retention_ai.models.predict import predict_churn
from customer_retention_ai.config.loader import get_customer_value, get_default_churn_probability

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
PROP_MODEL_PATH = BASE_DIR / "models" / "propensity_model.pkl"


def evaluate_policy(sample_size=None):
    """
    Fast policy evaluation using sampling.
    
    Args:
        sample_size: Number of customers to evaluate. If None, uses all (slower but more accurate)
    """
    try:
        # Load data
        df = pd.read_parquet(FEATURES_PATH)
        df = add_action_column(df)

        # Get predicted churn probabilities for each customer
        predictions = predict_churn()
        if predictions is None or len(predictions) == 0:
            raise ValueError("No churn predictions available")
        
        churn_prob_map = dict(zip(predictions["customer_id"], predictions["churn_probability"]))

        X = df.drop(columns=["customer_id", "feature_version", "churn", "action"], errors="ignore")
        actual_churn = df["churn"].values
        customer_ids = df["customer_id"].values

        customer_value = get_customer_value()
        
        # Default sample size for faster computation
        if sample_size is None:
            sample_size = min(1000, len(df))  # Sample 1000 or all if fewer
        
        # Random sampling for efficiency
        if len(df) > sample_size:
            indices = np.random.choice(len(df), sample_size, replace=False)
        else:
            indices = np.arange(len(df))

        ai_values = []
        baseline_values = []

        # Process customers (vectorized where possible)
        for idx in indices:
            try:
                row_features = X.iloc[[idx]]
                customer_id = customer_ids[idx]

                # Get actual churn for this customer from historical data
                actual_churned = bool(actual_churn[idx])

                # Get customer-specific uplift estimates
                uplift_estimates = estimate_uplift(row_features)

                # Use predicted churn probability
                predicted_churn_prob = churn_prob_map.get(customer_id, get_default_churn_probability())

                # AI recommends best action based on expected value
                decision = choose_best_action(
                    customer_id=customer_id,
                    churn_probability=predicted_churn_prob,
                    customer_value=customer_value,
                    uplift_estimates=uplift_estimates
                )

                ai_action = decision["recommended_action"]

                # Simulate the outcome of AI-recommended action
                if ai_action != "NO_ACTION":
                    uplift = uplift_estimates.get(ai_action, 0)
                    simulated_churned = actual_churned and np.random.random() > uplift
                else:
                    simulated_churned = actual_churned

                # Calculate values
                ai_outcome = 1 if not simulated_churned else 0
                ai_values.append(customer_value * ai_outcome)

                baseline_outcome = 1 if not actual_churned else 0
                baseline_values.append(customer_value * baseline_outcome)
            
            except Exception as e:
                # Continue processing other customers
                continue

        # Calculate final metrics
        if len(ai_values) == 0 or len(baseline_values) == 0:
            raise ValueError("No valid customer evaluations computed")
        
        ai_value = np.mean(ai_values)
        baseline_value = np.mean(baseline_values)
        
        # Avoid division by zero
        if baseline_value != 0:
            improvement = (ai_value - baseline_value) / baseline_value * 100
        else:
            improvement = 0

        return {
            "ai_policy_value": ai_value,
            "baseline_value": baseline_value,
            "improvement_pct": improvement
        }
        
    except Exception as e:
        print(f"Policy evaluation error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return default values on error
        return {
            "ai_policy_value": 0,
            "baseline_value": 0,
            "improvement_pct": 0
        }


if __name__ == "__main__":
    results = evaluate_policy()

    print("AI Policy Value:", results["ai_policy_value"])
    print("Baseline Value:", results["baseline_value"])
    print("Improvement %:", results["improvement_pct"])
