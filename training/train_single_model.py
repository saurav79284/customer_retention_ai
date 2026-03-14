# training/train_single_model.py

import lightgbm as lgb
import joblib
from pathlib import Path


def train_action_model(
    X,
    y,
    action_name: str,
    model_dir: Path
):
    model = lgb.LGBMClassifier(
    n_estimators=100,
    min_data_in_leaf=50,
    min_gain_to_split=0.0,
    verbosity=-1
)

    model.fit(X, y)

    model_path = model_dir / f"churn_{action_name.lower()}.pkl"
    joblib.dump(model, model_path)

    print(f"✅ Saved {model_path}")