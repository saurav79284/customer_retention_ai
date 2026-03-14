# features/build_features.py

import pandas as pd
from pathlib import Path

# =====================================================
# Feature Store Configuration
# =====================================================

FEATURE_VERSION = "v1.0.0"

RAW_DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
PROCESSED_DATA_PATH = Path("data/processed/features.parquet")


# =====================================================
# Feature Builder
# =====================================================

class FeatureBuilder:
    """
    Deterministic feature engineering for Telco Customer Churn dataset.
    Same logic used for training and inference.
    """

    def __init__(self, feature_version: str = FEATURE_VERSION):
        self.feature_version = feature_version

    def load_raw_data(self) -> pd.DataFrame:
        df = pd.read_csv(RAW_DATA_PATH)

        # Clean TotalCharges (sometimes blank)
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].replace(" ", pd.NA),
            errors="coerce"
        ).fillna(0.0)

        return df

    def build_features(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        features = pd.DataFrame()

        # ---------------- Identifiers ----------------
        features["customer_id"] = df["customerID"]
        features["feature_version"] = self.feature_version

        # ---------------- Numeric ----------------
        features["tenure"] = df["tenure"].astype(int)
        features["monthly_charges"] = df["MonthlyCharges"].astype(float)
        features["total_charges"] = df["TotalCharges"].astype(float)

        features["avg_monthly_spend"] = (
            features["total_charges"] / features["tenure"].replace(0, 1)
        )

        # ---------------- Binary Yes / No ----------------
        binary_cols = [
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling",
        ]

        for col in binary_cols:
            features[col.lower()] = (
                df[col]
                .replace({"Yes": 1, "No": 0})
                .astype(int)
            )

        # SeniorCitizen already 0/1, normalize name
        features["senior_citizen"] = df["SeniorCitizen"].astype(int)
        features.drop(columns=["seniorcitizen"], errors="ignore", inplace=True)

        # ---------------- One-Hot Encoded Categoricals ----------------
        categorical_cols = [
            "Contract",
            "PaymentMethod",
            "InternetService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        dummies = pd.get_dummies(
            df[categorical_cols],
            prefix=[c.lower() for c in categorical_cols],
            drop_first=False
        )

        features = pd.concat([features, dummies], axis=1)

        # ---------------- Target (Training Only) ----------------
        if training and "Churn" in df.columns:
            features["churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

        return features

    def save_features(self, df: pd.DataFrame):
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_DATA_PATH, index=False)

    def run(self, training: bool = True):
        df_raw = self.load_raw_data()
        df_features = self.build_features(df_raw, training=training)
        self.save_features(df_features)
        return df_features


# =====================================================
# CLI Entry
# =====================================================

if __name__ == "__main__":
    builder = FeatureBuilder()
    builder.run(training=True)