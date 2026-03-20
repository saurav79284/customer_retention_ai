import pandas as pd
from customer_retention_ai.evaluation.doubly_robust import doubly_robust_value


def test_doubly_robust():

    df = pd.DataFrame({
        "action": ["A","B","A","B"],
        "ai_action": ["A","A","A","B"],
        "propensity": [0.5,0.4,0.5,0.6],
        "reward": [10,20,30,40],
        "q_hat": [12,18,28,38]
    })

    value = doubly_robust_value(df)

    assert isinstance(value, float)
