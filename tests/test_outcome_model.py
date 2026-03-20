import pandas as pd
import numpy as np
from customer_retention_ai.evaluation.outcome_model import OutcomeModel



def test_outcome_model_runs():

    df = pd.DataFrame({
        "tenure": [5, 10, 20, 30],
        "monthly_charges": [50, 70, 80, 90],
        "action": ["NO_ACTION", "DISCOUNT_10", "LOYALTY_OFFER", "PRIORITY_SUPPORT"],
        "reward": [10, 20, 30, 40]
    })

    feature_cols = ["tenure", "monthly_charges"]

    model = OutcomeModel()

    model.fit(df, feature_cols)

    preds = model.predict(df)

    assert len(preds) == len(df)
