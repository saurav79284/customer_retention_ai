from decision_engine.expected_value import compute_expected_value

def test_positive_ev():
    ev = compute_expected_value(
        uplift=0.2,
        customer_value=1000,
        action_cost=100
    )
    assert ev == 100.0

def test_zero_ev():
    ev = compute_expected_value(
        uplift=0.1,
        customer_value=1000,
        action_cost=100
    )
    assert ev == 0.0

def test_negative_ev():
    ev = compute_expected_value(
        uplift=0.05,
        customer_value=1000,
        action_cost=100
    )
    assert ev < 0