import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


class TLearner:
    def __init__(self):
        self.models = {
            "NO_ACTION": joblib.load(MODEL_DIR / "churn_no_action.pkl"),
            "DISCOUNT_10": joblib.load(MODEL_DIR / "churn_discount_10.pkl"),
            "PRIORITY_SUPPORT": joblib.load(MODEL_DIR / "churn_priority_support.pkl"),
            "LOYALTY_OFFER": joblib.load(MODEL_DIR / "churn_loyalty_offer.pkl"),
        }

    def uplift(self, features: dict, action_id: str) -> float:
        """
        uplift = P(churn | no_action) − P(churn | action)
        """

        x = [list(features.values())]

        p_no = self.models["NO_ACTION"].predict_proba(x)[0][1]
        p_a = self.models[action_id].predict_proba(x)[0][1]

        return max(0.0, p_no - p_a)