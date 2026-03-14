import pandas as pd
from pathlib import Path


def load_telco_data(features_path: Path) -> pd.DataFrame:
    """
    Load training feature store.
    Assumes churn label is already present.
    """

    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    df = pd.read_parquet(features_path)

    if "churn" not in df.columns:
        raise ValueError(
            "Feature store does not contain 'churn'. "
            "Run FeatureBuilder with training=True."
        )

    return df