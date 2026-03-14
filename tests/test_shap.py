from explainability.shap_explainer import explain_predictions

def test_shap_output_format():
    results = explain_predictions(top_k=3)

    assert isinstance(results, list)
    assert "customer_id" in results[0]
    assert "top_drivers" in results[0]
    assert len(results[0]["top_drivers"]) == 3