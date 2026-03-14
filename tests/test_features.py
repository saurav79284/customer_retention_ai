import pandas as pd
from features.build_features import FeatureBuilder

def test_feature_builder_output_schema():
    builder = FeatureBuilder()
    df = builder.run(training=True)

    assert "customer_id" in df.columns
    assert "feature_version" in df.columns
    assert "churn" in df.columns
    assert len(df) > 0
    