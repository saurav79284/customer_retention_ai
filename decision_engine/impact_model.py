def impact(churn_probability: float, action_id: str) -> float:
    """
    Heuristic action-dependent uplift model.
    Returns expected churn reduction (0–1).
    """

    if action_id == "DISCOUNT_10":
        return 0.08 + 0.30 * churn_probability

    if action_id == "PRIORITY_SUPPORT":
        return 0.12 + 0.40 * churn_probability

    if action_id == "LOYALTY_OFFER":
        return 0.05 + 0.20 * churn_probability

    return 0.0