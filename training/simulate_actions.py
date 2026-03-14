import numpy as np
import pandas as pd


ACTIONS = [
    "NO_ACTION",
    "DISCOUNT_10",
    "PRIORITY_SUPPORT",
    "LOYALTY_OFFER"
]


def assign_action(row: pd.Series) -> str:
    """
    Synthetic treatment assignment WITHOUT using churn_probability.
    Uses tenure & spend as proxies (safe, available features).
    """

    tenure = row.get("tenure", 0)
    monthly = row.get("monthly_charges", 0.0)

    # High-value OR long-tenure customers
    if tenure >= 24 or monthly >= 80:
        return np.random.choice(
            ["DISCOUNT_10", "PRIORITY_SUPPORT"],
            p=[0.6, 0.4]
        )

    # Medium customers
    if tenure >= 12 or monthly >= 50:
        return np.random.choice(
            ["LOYALTY_OFFER", "DISCOUNT_10"],
            p=[0.6, 0.4]
        )

    # Low-value / new customers
    return "NO_ACTION"


def add_action_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a single 'action' column.
    Does NOT modify or drop any existing columns.
    """

    if "action" in df.columns:
        raise ValueError("'action' column already exists")

    df = df.copy()
    df["action"] = df.apply(assign_action, axis=1)

    return df