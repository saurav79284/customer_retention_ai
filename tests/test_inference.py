from models.predict import predict_churn

def test_inference_runs():
    df = predict_churn()

    assert "customer_id" in df.columns
    assert "churn_probability" in df.columns
    assert df["churn_probability"].between(0, 1).all()