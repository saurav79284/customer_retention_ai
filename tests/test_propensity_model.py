import pandas as pd
from customer_retention_ai.evaluation.propensity_model import PropensityModel


def test_propensity_model():

    df = pd.DataFrame({
        "tenure": [5,10,15,20],
        "monthly_charges": [40,60,80,100],
        "action": ["NO_ACTION","DISCOUNT_10","DISCOUNT_10","LOYALTY_OFFER"]
    })

    feature_cols = ["tenure","monthly_charges"]

    model = PropensityModel()

    model.fit(df, feature_cols)

    probs = model.predict_proba(df)

    assert probs.shape[0] == len(df)
