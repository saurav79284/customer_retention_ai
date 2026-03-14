from typing import Dict, Any, List
from customer_retention_ai.decision_engine.actions import get_available_actions, Action
from customer_retention_ai.decision_engine.expected_value import compute_expected_value


def choose_best_action(
    customer_id: str,
    churn_probability: float,
    customer_value: float,
    uplift_estimates: Dict[str, float]
) -> Dict[str, Any]:
    """
    Selects the action with the highest positive Expected Value.
    """

    actions: List[Action] = get_available_actions()
    evaluations = []

    for action in actions:
        assert action.action_id in uplift_estimates, (
        f"Missing uplift for action: {action.action_id}"
        )

        uplift = uplift_estimates[action.action_id]

        ev = compute_expected_value(
            uplift=uplift,
            customer_value=customer_value,
            action_cost=action.cost
        )

        evaluations.append({
            "action_id": action.action_id,
            "expected_value": round(ev, 2)
        })

    best = max(evaluations, key=lambda x: x["expected_value"])

    if best["expected_value"] <= 0:
        return {
            "customer_id": customer_id,
            "recommended_action": "NO_ACTION",
            "expected_value": 0.0,
            "reason": "No action has positive expected value",
            "all_action_evaluations": evaluations
        }

    return {
        "customer_id": customer_id,
        "recommended_action": best["action_id"],
        "expected_value": best["expected_value"],
        "reason": "Highest positive expected value",
        "all_action_evaluations": evaluations
    }