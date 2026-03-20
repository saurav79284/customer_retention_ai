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
    Selects action based on risk-aware strategy:
    - HIGH-RISK (churn > 50%): Recommend best action with positive uplift (proactive retention)
    - MED-RISK (30% < churn <= 50%): Recommend if EV > 0 and uplift > 5%
    - LOW-RISK (churn <= 30%): Recommend only if EV > 0 (conservative)
    
    This balances cost efficiency with retention urgency.
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
            action_cost=action.cost,
            churn_probability=churn_probability
        )

        evaluations.append({
            "action_id": action.action_id,
            "uplift": uplift,
            "expected_value": round(ev, 2)
        })

    # Sort by EV descending
    evaluations = sorted(evaluations, key=lambda x: x["expected_value"], reverse=True)
    best = evaluations[0]

    # ===== RISK-AWARE DECISION LOGIC =====
    
    # HIGH-RISK customers: Recommend if ANY positive-uplift action available (be proactive)
    if churn_probability > 0.5:
        positive_uplift_actions = [e for e in evaluations if e["uplift"] > 0.01]
        if positive_uplift_actions:
            best = max(positive_uplift_actions, key=lambda x: x["expected_value"])
            return {
                "customer_id": customer_id,
                "recommended_action": best["action_id"],
                "expected_value": best["expected_value"],
                "reason": f"High-risk customer ({churn_probability:.1%} churn) - proactive retention",
                "all_action_evaluations": evaluations
            }
    
    # MED-RISK customers: Recommend if EV > 0 AND uplift > 5% (selective)
    if 0.3 < churn_probability <= 0.5:
        qualified_actions = [e for e in evaluations if e["expected_value"] > 0 and e["uplift"] > 0.05]
        if qualified_actions:
            best = qualified_actions[0]
            return {
                "customer_id": customer_id,
                "recommended_action": best["action_id"],
                "expected_value": best["expected_value"],
                "reason": f"Medium-risk customer ({churn_probability:.1%} churn) - if strong uplift",
                "all_action_evaluations": evaluations
            }
    
    # LOW-RISK customers: Recommend only if EV > 0 (conservative)
    if best["expected_value"] > 0:
        return {
            "customer_id": customer_id,
            "recommended_action": best["action_id"],
            "expected_value": best["expected_value"],
            "reason": f"Low-risk customer ({churn_probability:.1%} churn) - high EV required",
            "all_action_evaluations": evaluations
        }

    # Default: NO_ACTION
    return {
        "customer_id": customer_id,
        "recommended_action": "NO_ACTION",
        "expected_value": 0.0,
        "reason": "No action meets risk-appropriate criteria",
        "all_action_evaluations": evaluations
    }