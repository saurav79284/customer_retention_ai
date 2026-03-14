import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from customer_retention_ai.training.load_data import load_telco_data
from customer_retention_ai.training.simulate_actions import add_action_column

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    df = load_telco_data(FEATURES_PATH)
    df = add_action_column(df)

    X = df.drop(columns=["customer_id", "feature_version", "churn", "action"])
    y = df["action"]

    model = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000
    )

    model.fit(X, y)

    joblib.dump(model, MODEL_DIR / "propensity_model.pkl")
    print("✅ Saved propensity_model.pkl")


if __name__ == "__main__":
    main()