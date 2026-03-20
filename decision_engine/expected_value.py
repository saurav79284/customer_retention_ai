def compute_expected_value(
    uplift: float,
    customer_value: float,
    action_cost: float,
    churn_probability: float = 0.5
) -> float:
    """
    EV = (uplift × customer_value) - action_cost
    
    Calculates expected value of intervention.
    
    IMPORTANT: Do NOT multiply uplift by churn_probability!
    - Uplift is ALREADY a % churn reduction
    - It represents: (churn_with_action - churn_without_action)
    - Multiplying by churn_probability = DOUBLE-PENALTY for low-risk customers
    - Uplift % × Customer Value = $ benefit from intervention
    
    Examples:
    - Customer at 5% churn, 16% uplift: EV = (0.16 × 1200) - 20 = $172 (not broken down by churn)
    - Customer at 80% churn, 16% uplift: EV = (0.16 × 1200) - 20 = $172 (same calculation!)
    - Why: The 16% uplift already IS their personalized benefit
    """
    # Direct calculation: uplift as % of customer value, minus cost
    benefit = uplift * customer_value
    return benefit - action_cost