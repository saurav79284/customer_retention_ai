from customer_retention_ai.decision_engine.policy import choose_best_action


def test_best_action_selected():
    result = choose_best_action(
        customer_id="C1",
        churn_probability=0.6,
        customer_value=1000,
        uplift_estimates={
            "DISCOUNT_10": 0.10,
            "PRIORITY_SUPPORT": 0.50,
            "LOYALTY_OFFER": 0.05
        }
    )

    assert result["recommended_action"] == "PRIORITY_SUPPORT"
    assert result["expected_value"] == 470