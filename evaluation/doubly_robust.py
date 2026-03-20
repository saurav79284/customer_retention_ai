import numpy as np
import pandas as pd


def doubly_robust_value(
    df: pd.DataFrame,
    action_col="action",
    ai_action_col="ai_action",
    propensity_col="propensity",
    reward_col="reward",
    q_hat_col="q_hat"
):
    """
    Doubly Robust Policy Evaluation.

    df must contain:
    - action (historical action)
    - ai_action (AI recommended action)
    - propensity (P(action | features))
    - reward (observed reward)
    - q_hat (predicted reward model)
    """

    match = (df[action_col] == df[ai_action_col]).astype(int)

    correction = match * (
        df[reward_col] - df[q_hat_col]
    ) / df[propensity_col]

    dr_estimate = df[q_hat_col] + correction

    return dr_estimate.mean()
