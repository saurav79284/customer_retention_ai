# decision_engine/ab_test_simulator.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import stats

from customer_retention_ai.decision_engine.policy import choose_best_action
from customer_retention_ai.decision_engine.expected_value import compute_expected_value
from customer_retention_ai.models.predict import predict_churn

# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
MODEL_DIR = BASE_DIR / "models"
PROPENSITY_MODEL_PATH = MODEL_DIR / "propensity_model.pkl"

# =====================================================
# Configuration
# =====================================================

DEFAULT_CUSTOMER_VALUE = 1200
DEFAULT_CAMPAIGN_SIZE = 1000
RANDOM_STATE = 42


# =====================================================
# Outcome Simulation
# =====================================================

class OutcomeSimulator:
    """
    Simulates customer churn outcomes under different actions.
    Based on historical uplift estimates from T-Learner.
    """

    # Realistic uplift rates from historical campaign analysis
    # Based on 7,043 customer dataset analysis
    DEFAULT_UPLIFT_EFFECTS = {
        "NO_ACTION": {"uplift": 0.0, "response_rate": 0.0},
        "DISCOUNT_10": {"uplift": 0.12, "response_rate": 0.65},      # 12% churn reduction
        "PRIORITY_SUPPORT": {"uplift": 0.11, "response_rate": 0.40}, # 11% churn reduction
        "LOYALTY_OFFER": {"uplift": 0.16, "response_rate": 0.30},    # 16% churn reduction
    }

    @staticmethod
    def simulate_customer_outcome(
        base_churn_prob: float,
        action: str,
        uplift_effects: dict = None,
        seed: int = None
    ) -> dict:
        """
        Simulate whether customer churns given an action.
        
        Returns:
            {
                "churned": bool,
                "responded_to_action": bool,
                "outcome": 1 if retained, 0 if churned
            }
        """

        if seed is not None:
            np.random.seed(seed)

        if uplift_effects is None:
            uplift_effects = OutcomeSimulator.DEFAULT_UPLIFT_EFFECTS

        effect = uplift_effects.get(action, {})
        uplift = effect.get("uplift", 0.0)
        response_rate = effect.get("response_rate", 0.0)

        # Adjusted churn probability after action
        adjusted_churn_prob = max(0.0, base_churn_prob - uplift)

        # Does customer respond to action?
        responded = np.random.random() < response_rate if action != "NO_ACTION" else False

        # Churn outcome
        churned = np.random.random() < adjusted_churn_prob

        return {
            "churned": churned,
            "responded_to_action": responded,
            "outcome": 0 if churned else 1  # 1 = retained
        }


# =====================================================
# A/B Test Simulator
# =====================================================

class ABTestSimulator:
    """
    Simulates A/B test outcomes for the retention policy.
    Compares AI-recommended actions vs baseline (no action).
    """

    def __init__(
        self,
        features_path: Path = FEATURES_PATH,
        propensity_model_path: Path = PROPENSITY_MODEL_PATH,
        customer_value: float = DEFAULT_CUSTOMER_VALUE,
        random_state: int = RANDOM_STATE
    ):
        """
        Initialize simulator with data and models.
        
        Args:
            features_path: Path to feature store
            propensity_model_path: Path to propensity model
            customer_value: Customer lifetime value
            random_state: For reproducibility
        """

        self.features_path = features_path
        self.propensity_model_path = propensity_model_path
        self.customer_value = customer_value
        self.random_state = random_state

        np.random.seed(random_state)

        self.features_df = None
        self.propensity_model = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load feature store and propensity model."""

        if not self.features_path.exists():
            raise FileNotFoundError(f"Features not found: {self.features_path}")

        self.features_df = pd.read_parquet(self.features_path)

        if self.propensity_model_path.exists():
            self.propensity_model = joblib.load(self.propensity_model_path)
        else:
            self.propensity_model = None

    def sample_customers(self, n_customers: int = DEFAULT_CAMPAIGN_SIZE):
        """Random sample of customers for simulation."""

        if n_customers > len(self.features_df):
            n_customers = len(self.features_df)

        return self.features_df.sample(n=n_customers, random_state=self.random_state)

    def simulate_campaign(
        self,
        sample_df: pd.DataFrame,
        uplift_estimates: dict = None
    ) -> pd.DataFrame:
        """
        Simulate outcomes for treatment and control groups.
        
        Args:
            sample_df: Sample of customers
            uplift_estimates: Uplift for each action (override defaults)
        
        Returns:
            DataFrame with: customer_id, ai_action, baseline_action, 
            churn_prob, ai_reward, baseline_reward, action_cost
        """

        if uplift_estimates is None:
            uplift_estimates = {
                "DISCOUNT_10": 0.12,         # Realistic: 12%
                "PRIORITY_SUPPORT": 0.11,   # Realistic: 11%
                "LOYALTY_OFFER": 0.16       # Realistic: 16%
            }

        # Get predicted churn probabilities for sampled customers
        try:
            all_predictions = predict_churn()
            churn_prob_map = dict(zip(all_predictions["customer_id"], all_predictions["churn_probability"]))
        except Exception as e:
            print(f"Warning: Could not load churn predictions: {e}")
            churn_prob_map = {}

        # Convert uplift_estimates to uplift_effects format for outcome simulation
        uplift_effects = OutcomeSimulator.DEFAULT_UPLIFT_EFFECTS.copy()
        for action, uplift in uplift_estimates.items():
            if action in uplift_effects:
                uplift_effects[action]["uplift"] = uplift

        results = []

        for idx, row in sample_df.iterrows():
            customer_id = row["customer_id"]
            
            # Use actual predicted churn probability (not hardcoded 0.5!)
            base_churn_prob = churn_prob_map.get(customer_id, 0.5)

            # AI Policy: choose_best_action with ACTUAL churn probability
            ai_decision = choose_best_action(
                customer_id=customer_id,
                churn_probability=base_churn_prob,
                customer_value=self.customer_value,
                uplift_estimates=uplift_estimates
            )

            ai_action = ai_decision["recommended_action"]
            ai_ev = ai_decision["expected_value"]

            # Baseline: no action
            baseline_action = "NO_ACTION"
            baseline_ev = 0.0

            # Simulate outcomes with current uplift_effects
            ai_outcome = OutcomeSimulator.simulate_customer_outcome(
                base_churn_prob,
                ai_action,
                uplift_effects=uplift_effects,
                seed=self.random_state + idx
            )

            baseline_outcome = OutcomeSimulator.simulate_customer_outcome(
                base_churn_prob,
                baseline_action,
                uplift_effects=uplift_effects,
                seed=self.random_state + idx + 10000
            )

            # Rewards: customer_value if retained, 0 if churned
            ai_reward = self.customer_value * ai_outcome["outcome"]
            baseline_reward = self.customer_value * baseline_outcome["outcome"]

            # Action cost (lookup from policy)
            action_costs = {
                "DISCOUNT_10": 50.0,
                "PRIORITY_SUPPORT": 30.0,
                "LOYALTY_OFFER": 20.0,
                "NO_ACTION": 0.0
            }
            action_cost = action_costs.get(ai_action, 0.0)

            results.append({
                "customer_id": customer_id,
                "ai_action": ai_action,
                "baseline_action": baseline_action,
                "base_churn_prob": base_churn_prob,
                "ai_outcome": ai_outcome["outcome"],
                "baseline_outcome": baseline_outcome["outcome"],
                "ai_reward": ai_reward,
                "baseline_reward": baseline_reward,
                "action_cost": action_cost,
                "ai_net_benefit": ai_reward - action_cost,
                "baseline_net_benefit": baseline_reward,
                "responded_to_action": ai_outcome["responded_to_action"]
            })

        return pd.DataFrame(results)

    def compute_metrics(self, results_df: pd.DataFrame) -> dict:
        """
        Compute campaign metrics and statistical significance.
        
        Args:
            results_df: Output from simulate_campaign()
        
        Returns:
            {
                "ai_total_revenue": float,
                "baseline_total_revenue": float,
                "ai_total_cost": float,
                "ai_net_revenue": float,
                "baseline_net_revenue": float,
                "roi_percentage": float,
                "revenue_lift_percentage": float,
                "retention_rate_ai": float,
                "retention_rate_baseline": float,
                "avg_eu_per_customer_ai": float,
                "avg_eu_per_customer_baseline": float,
                "p_value": float,
                "statistically_significant": bool,
                "sample_size": int
            }
        """

        n = len(results_df)

        # Revenue metrics
        ai_total_revenue = results_df["ai_reward"].sum()
        baseline_total_revenue = results_df["baseline_reward"].sum()
        ai_total_cost = results_df["action_cost"].sum()

        ai_net_revenue = ai_total_revenue - ai_total_cost
        baseline_net_revenue = baseline_total_revenue

        # ROI
        roi_pct = (
            (ai_net_revenue - baseline_net_revenue) / baseline_net_revenue * 100
            if baseline_net_revenue > 0
            else 0.0
        )

        # Lift
        revenue_lift_pct = (
            (ai_total_revenue - baseline_total_revenue) / baseline_total_revenue * 100
            if baseline_total_revenue > 0
            else 0.0
        )

        # Retention rates
        retention_rate_ai = results_df["ai_outcome"].mean()
        retention_rate_baseline = results_df["baseline_outcome"].mean()

        # Per-customer metrics
        avg_eu_ai = ai_net_revenue / n if n > 0 else 0.0
        avg_eu_baseline = baseline_net_revenue / n if n > 0 else 0.0

        # Statistical significance (paired t-test on net benefits)
        # Compare AI policy net revenue vs baseline net revenue
        ai_net_benefits = results_df["ai_net_benefit"].values
        baseline_net_benefits = results_df["baseline_net_benefit"].values

        # Use paired t-test since we're comparing same customers under two conditions
        # This is more appropriate than independent samples t-test
        t_stat, p_value = stats.ttest_rel(ai_net_benefits, baseline_net_benefits)
        significant = p_value < 0.05

        return {
            "ai_total_revenue": round(ai_total_revenue, 2),
            "baseline_total_revenue": round(baseline_total_revenue, 2),
            "ai_total_cost": round(ai_total_cost, 2),
            "ai_net_revenue": round(ai_net_revenue, 2),
            "baseline_net_revenue": round(baseline_net_revenue, 2),
            "roi_percentage": round(roi_pct, 2),
            "revenue_lift_percentage": round(revenue_lift_pct, 2),
            "retention_rate_ai": round(retention_rate_ai, 4),
            "retention_rate_baseline": round(retention_rate_baseline, 4),
            "avg_eu_per_customer_ai": round(avg_eu_ai, 2),
            "avg_eu_per_customer_baseline": round(avg_eu_baseline, 2),
            "p_value": round(p_value, 4),
            "statistically_significant": significant,
            "sample_size": n,
            "t_statistic": round(t_stat, 4)
        }

    def run_simulation(
        self,
        n_customers: int = DEFAULT_CAMPAIGN_SIZE,
        uplift_estimates: dict = None
    ) -> dict:
        """
        Run end-to-end simulation: sample → campaign → metrics.
        
        Args:
            n_customers: Campaign size
            uplift_estimates: Override default uplift estimates
        
        Returns:
            {
                "campaign_results": DataFrame,
                "metrics": dict,
                "action_distribution": dict
            }
        """

        # Sample customers
        sample = self.sample_customers(n_customers)

        # Simulate campaign
        results = self.simulate_campaign(sample, uplift_estimates)

        # Compute metrics
        metrics = self.compute_metrics(results)

        # Action distribution
        action_dist = results["ai_action"].value_counts().to_dict()

        return {
            "campaign_results": results,
            "metrics": metrics,
            "action_distribution": action_dist
        }


# =====================================================
# Convenience Functions
# =====================================================

def quick_simulation(
    n_customers: int = 1000,
    customer_value: float = DEFAULT_CUSTOMER_VALUE
) -> dict:
    """
    Quick simulation with default settings.
    
    Args:
        n_customers: Campaign size
        customer_value: Customer lifetime value
    
    Returns:
        Simulation results with metrics
    """

    simulator = ABTestSimulator(customer_value=customer_value)
    return simulator.run_simulation(n_customers=n_customers)


if __name__ == "__main__":
    # Example usage
    print("🚀 Starting A/B Test Simulation...\n")

    results = quick_simulation(n_customers=500)

    print("=" * 60)
    print("CAMPAIGN RESULTS")
    print("=" * 60)

    metrics = results["metrics"]

    print(f"\n📊 Sample Size: {metrics['sample_size']} customers")
    print(f"\nRevenue Metrics:")
    print(f"  AI Total Revenue:         ${metrics['ai_total_revenue']:,.2f}")
    print(f"  Baseline Total Revenue:   ${metrics['baseline_total_revenue']:,.2f}")
    print(f"  AI Campaign Cost:         ${metrics['ai_total_cost']:,.2f}")
    print(f"  AI Net Revenue:           ${metrics['ai_net_revenue']:,.2f}")
    print(f"  Baseline Net Revenue:     ${metrics['baseline_net_revenue']:,.2f}")

    print(f"\n🎯 Performance Metrics:")
    print(f"  Revenue Lift:             {metrics['revenue_lift_percentage']:+.2f}%")
    print(f"  ROI:                      {metrics['roi_percentage']:+.2f}%")
    print(f"  Avg EU per Customer (AI): ${metrics['avg_eu_per_customer_ai']:.2f}")
    print(f"  Avg EU per Customer (BL): ${metrics['avg_eu_per_customer_baseline']:.2f}")

    print(f"\n📈 Retention Rates:")
    print(f"  AI Policy:                {metrics['retention_rate_ai']:.2%}")
    print(f"  Baseline (No Action):     {metrics['retention_rate_baseline']:.2%}")

    print(f"\n✅ Statistical Significance:")
    print(f"  P-value:                  {metrics['p_value']:.4f}")
    print(f"  Significant (α=0.05):     {metrics['statistically_significant']}")

    print(f"\n📋 Action Distribution:")
    for action, count in results["action_distribution"].items():
        pct = count / metrics["sample_size"] * 100
        print(f"  {action}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
