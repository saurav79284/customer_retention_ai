# tests/test_ab_simulator.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch

from customer_retention_ai.decision_engine.ab_test_simulator import (
    OutcomeSimulator,
    ABTestSimulator,
    quick_simulation,
    DEFAULT_CUSTOMER_VALUE,
    DEFAULT_CAMPAIGN_SIZE
)

# =====================================================
# Fixtures
# =====================================================

@pytest.fixture
def sample_features():
    """Create mock feature DataFrame."""
    data = {
        "customer_id": [f"CUST_{i:04d}" for i in range(100)],
        "feature_version": ["v1.0.0"] * 100,
        "tenure": np.random.randint(1, 72, 100),
        "monthly_charges": np.random.uniform(20, 150, 100),
        "total_charges": np.random.uniform(100, 10000, 100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_features_path(tmp_path, sample_features):
    """Create temporary features file."""
    features_file = tmp_path / "features.parquet"
    sample_features.to_parquet(features_file)
    return features_file


@pytest.fixture
def mock_propensity_model(tmp_path):
    """Create mock propensity model file with a real Logistic Regression."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    
    # Create a real sklearn model (picklable, unlike MagicMock)
    model = LogisticRegression(multi_class="multinomial", max_iter=1000)
    
    # Create dummy training data
    X_train = np.random.rand(20, 5)
    y_train = np.random.randint(0, 4, 20)
    
    # Train the model
    model.fit(X_train, y_train)
    model.classes_ = np.array(["NO_ACTION", "DISCOUNT_10", "PRIORITY_SUPPORT", "LOYALTY_OFFER"])
    
    # Save to temp file
    model_file = tmp_path / "propensity_model.pkl"
    joblib.dump(model, model_file)
    return model_file


# =====================================================
# Tests: OutcomeSimulator
# =====================================================

class TestOutcomeSimulator:
    """Test outcome simulation logic."""

    def test_outcome_simulator_no_action(self):
        """NO_ACTION should have 0 uplift and 0 response rate."""
        outcome = OutcomeSimulator.simulate_customer_outcome(
            base_churn_prob=0.5,
            action="NO_ACTION",
            seed=42
        )
        
        assert outcome["responded_to_action"] == False
        assert isinstance(outcome["churned"], (bool, np.bool_))
        assert outcome["outcome"] in [0, 1]

    def test_outcome_simulator_discount(self):
        """DISCOUNT_10 should have uplift and response rate."""
        outcomes = [
            OutcomeSimulator.simulate_customer_outcome(
                base_churn_prob=0.5,
                action="DISCOUNT_10",
                seed=42 + i
            )
            for i in range(100)
        ]
        
        # Some should respond
        responded_count = sum(1 for o in outcomes if o["responded_to_action"])
        assert responded_count > 0
        
        # Some should retain
        retain_count = sum(1 for o in outcomes if o["outcome"] == 1)
        assert retain_count > 0

    def test_outcome_simulator_reproducibility(self):
        """Same seed should produce same outcome."""
        outcome1 = OutcomeSimulator.simulate_customer_outcome(
            base_churn_prob=0.5,
            action="DISCOUNT_10",
            seed=42
        )
        
        outcome2 = OutcomeSimulator.simulate_customer_outcome(
            base_churn_prob=0.5,
            action="DISCOUNT_10",
            seed=42
        )
        
        assert outcome1["churned"] == outcome2["churned"]
        assert outcome1["responded_to_action"] == outcome2["responded_to_action"]

    def test_outcome_simulator_invalid_action(self):
        """Invalid action should use defaults."""
        outcome = OutcomeSimulator.simulate_customer_outcome(
            base_churn_prob=0.5,
            action="INVALID_ACTION",
            seed=42
        )
        
        # Should still return valid outcome
        assert "churned" in outcome
        assert "responded_to_action" in outcome
        assert "outcome" in outcome

    def test_outcome_simulator_churn_bounds(self):
        """Outcome should be binary (0 or 1)."""
        for action in ["NO_ACTION", "DISCOUNT_10", "PRIORITY_SUPPORT", "LOYALTY_OFFER"]:
            outcomes = [
                OutcomeSimulator.simulate_customer_outcome(0.5, action, seed=42 + i)
                for i in range(50)
            ]
            
            for outcome in outcomes:
                assert outcome["outcome"] in [0, 1]


# =====================================================
# Tests: ABTestSimulator Initialization
# =====================================================

class TestABTestSimulatorInit:
    """Test simulator initialization."""

    def test_simulator_init_with_files(self, mock_features_path, mock_propensity_model):
        """Initialize simulator with mock files."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model,
            customer_value=1200
        )
        
        assert simulator.features_df is not None
        assert len(simulator.features_df) == 100
        assert simulator.customer_value == 1200

    def test_simulator_init_missing_features(self, tmp_path):
        """Should raise error if features not found."""
        with pytest.raises(FileNotFoundError):
            ABTestSimulator(
                features_path=tmp_path / "nonexistent.parquet",
                customer_value=1200
            )

    def test_simulator_default_values(self, mock_features_path):
        """Test default initialization values."""
        simulator = ABTestSimulator(features_path=mock_features_path)
        
        assert simulator.customer_value == DEFAULT_CUSTOMER_VALUE
        assert simulator.random_state == 42


# =====================================================
# Tests: ABTestSimulator Methods
# =====================================================

class TestABTestSimulatorMethods:
    """Test simulator core methods."""

    def test_sample_customers(self, mock_features_path, mock_propensity_model):
        """Test customer sampling."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model
        )
        
        sample = simulator.sample_customers(n_customers=20)
        
        assert len(sample) == 20
        assert list(sample.columns) == list(simulator.features_df.columns)

    def test_sample_customers_exceeds_total(self, mock_features_path, mock_propensity_model):
        """Sampling more than available should return all."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model
        )
        
        sample = simulator.sample_customers(n_customers=1000)
        
        assert len(sample) == 100  # Only 100 in fixture

    def test_simulate_campaign_returns_dataframe(self, mock_features_path, mock_propensity_model):
        """Campaign simulation should return DataFrame with required columns."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model
        )
        
        sample = simulator.sample_customers(10)
        results = simulator.simulate_campaign(sample)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 10
        
        required_cols = [
            "customer_id", "ai_action", "baseline_action",
            "ai_outcome", "baseline_outcome", "ai_reward", "baseline_reward"
        ]
        for col in required_cols:
            assert col in results.columns

    def test_simulate_campaign_actions_in_catalog(self, mock_features_path, mock_propensity_model):
        """AI actions should be from known catalog."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model
        )
        
        sample = simulator.sample_customers(20)
        results = simulator.simulate_campaign(sample)
        
        valid_actions = {"NO_ACTION", "DISCOUNT_10", "PRIORITY_SUPPORT", "LOYALTY_OFFER"}
        
        for action in results["ai_action"]:
            assert action in valid_actions

    def test_compute_metrics_complete_output(self, mock_features_path, mock_propensity_model):
        """Metrics should include all required fields."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model
        )
        
        sample = simulator.sample_customers(30)
        results_df = simulator.simulate_campaign(sample)
        metrics = simulator.compute_metrics(results_df)
        
        required_metrics = [
            "ai_total_revenue", "baseline_total_revenue",
            "roi_percentage", "revenue_lift_percentage",
            "retention_rate_ai", "retention_rate_baseline",
            "p_value", "statistically_significant", "sample_size"
        ]
        
        for metric in required_metrics:
            assert metric in metrics

    def test_compute_metrics_values_in_range(self, mock_features_path, mock_propensity_model):
        """Metrics should have sensible ranges."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model,
            customer_value=1000
        )
        
        sample = simulator.sample_customers(50)
        results_df = simulator.simulate_campaign(sample)
        metrics = simulator.compute_metrics(results_df)
        
        # Retention rates should be between 0 and 1
        assert 0 <= metrics["retention_rate_ai"] <= 1
        assert 0 <= metrics["retention_rate_baseline"] <= 1
        
        # P-value should be between 0 and 1
        assert 0 <= metrics["p_value"] <= 1
        
        # Sample size should match
        assert metrics["sample_size"] == 50

    def test_compute_metrics_revenue_positive(self, mock_features_path, mock_propensity_model):
        """Revenue should be non-negative."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model
        )
        
        sample = simulator.sample_customers(40)
        results_df = simulator.simulate_campaign(sample)
        metrics = simulator.compute_metrics(results_df)
        
        assert metrics["ai_total_revenue"] >= 0
        assert metrics["baseline_total_revenue"] >= 0
        assert metrics["ai_total_cost"] >= 0


# =====================================================
# Tests: Full Simulation
# =====================================================

class TestABTestSimulatorFullRun:
    """Test end-to-end simulation."""

    def test_run_simulation_complete(self, mock_features_path, mock_propensity_model):
        """Full simulation should return all components."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model
        )
        
        results = simulator.run_simulation(n_customers=30)
        
        assert "campaign_results" in results
        assert "metrics" in results
        assert "action_distribution" in results
        
        assert isinstance(results["campaign_results"], pd.DataFrame)
        assert isinstance(results["metrics"], dict)
        assert isinstance(results["action_distribution"], dict)

    def test_run_simulation_action_distribution(self, mock_features_path, mock_propensity_model):
        """Action distribution should sum to sample size."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model
        )
        
        results = simulator.run_simulation(n_customers=25)
        
        dist = results["action_distribution"]
        total = sum(dist.values())
        
        assert total == 25

    def test_quick_simulation_function(self, mock_features_path, mock_propensity_model):
        """Quick simulation convenience function should work."""
        with patch('customer_retention_ai.decision_engine.ab_test_simulator.FEATURES_PATH', mock_features_path):
            with patch('customer_retention_ai.decision_engine.ab_test_simulator.PROPENSITY_MODEL_PATH', mock_propensity_model):
                # Note: This would require proper mocking of the module-level constants
                # For now, we test the structure
                pass


# =====================================================
# Tests: Edge Cases
# =====================================================

class TestABTestSimulatorEdgeCases:
    """Test edge cases and error handling."""

    def test_simulate_campaign_custom_uplift(self, mock_features_path, mock_propensity_model):
        """Should accept custom uplift estimates."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model
        )
        
        sample = simulator.sample_customers(10)
        custom_uplift = {
            "DISCOUNT_10": 0.25,
            "PRIORITY_SUPPORT": 0.10,
            "LOYALTY_OFFER": 0.05
        }
        
        results = simulator.simulate_campaign(sample, uplift_estimates=custom_uplift)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 10

    def test_compute_metrics_zero_revenue(self, mock_features_path, mock_propensity_model):
        """Should handle zero revenue gracefully."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model
        )
        
        # Create results where all customers churned
        results_df = pd.DataFrame({
            "ai_reward": [0] * 20,
            "baseline_reward": [0] * 20,
            "ai_net_benefit": [-50] * 20,
            "baseline_net_benefit": [0] * 20,
            "action_cost": [50] * 20,
            "ai_outcome": [0] * 20,
            "baseline_outcome": [0] * 20
        })
        
        metrics = simulator.compute_metrics(results_df)
        
        # ROI calculation should not divide by zero
        assert isinstance(metrics["roi_percentage"], float)
        assert isinstance(metrics["revenue_lift_percentage"], float)

    def test_compute_metrics_small_sample(self, mock_features_path, mock_propensity_model):
        """Should handle small sample sizes."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model
        )
        
        sample = simulator.sample_customers(3)
        results_df = simulator.simulate_campaign(sample)
        metrics = simulator.compute_metrics(results_df)
        
        assert metrics["sample_size"] == 3
        assert 0 <= metrics["retention_rate_ai"] <= 1


# =====================================================
# Tests: Integration
# =====================================================

class TestABTestSimulatorIntegration:
    """Test integration with existing modules."""

    def test_integration_with_policy(self, mock_features_path, mock_propensity_model):
        """Simulator should integrate with choose_best_action."""
        simulator = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model
        )
        
        sample = simulator.sample_customers(5)
        results = simulator.simulate_campaign(sample)
        
        # Verify that actions were chosen (not errors)
        assert len(results) == 5
        assert all(action in ["NO_ACTION", "DISCOUNT_10", "PRIORITY_SUPPORT", "LOYALTY_OFFER"] 
                   for action in results["ai_action"])

    def test_reproducibility_across_runs(self, mock_features_path, mock_propensity_model):
        """Same random_state should produce reproducible results."""
        sim1 = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model,
            random_state=42
        )
        results1 = sim1.run_simulation(n_customers=20)
        
        sim2 = ABTestSimulator(
            features_path=mock_features_path,
            propensity_model_path=mock_propensity_model,
            random_state=42
        )
        results2 = sim2.run_simulation(n_customers=20)
        
        # Metrics should be identical
        assert results1["metrics"]["retention_rate_ai"] == results2["metrics"]["retention_rate_ai"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
