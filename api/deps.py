import json
import joblib
from pathlib import Path

MODEL_DIR = Path("models/artifacts")
MODEL_PATH = MODEL_DIR / "churn_lightgbm.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def load_model():
    return joblib.load(MODEL_PATH)


def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)