def compute_expected_value(
    uplift: float,
    customer_value: float,
    action_cost: float
) -> float:
    """
    EV = uplift × customer_value − action_cost
    """
    return uplift * customer_value - action_cost