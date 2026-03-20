import numpy as np
import pandas as pd


def ips_policy_value(
    df: pd.DataFrame,
    action_col="action",
    ai_action_col="ai_action",
    propensity_col="propensity",
    reward_col="reward"
):
    """
    Estimate value of AI policy using IPS.

    df must contain:
    - historical action
    - AI recommended action
    - propensity score
    - reward
    """

    mask = df[action_col] == df[ai_action_col]

    weights = 1.0 / df[propensity_col]

    weighted_reward = mask * weights * df[reward_col]

    return weighted_reward.mean()
