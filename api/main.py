from fastapi import FastAPI
import uuid

from api.schemas import PredictRequest, PredictResponse, DecisionResponse
from models.predict import predict_churn
from explainability.shap_explainer import explain_predictions
from decision_engine.policy import choose_best_action

app = FastAPI(
    title="Customer Retention Decision Intelligence API",
    version="1.0.0"
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    preds = predict_churn()
    row = preds.iloc[0]

    return PredictResponse(
        request_id=row["request_id"],
        customer_id=row["customer_id"],
        churn_probability=float(row["churn_probability"]),
        model_version=row["model_version"],
        feature_version=row["feature_version"],
    )


@app.post("/decision", response_model=DecisionResponse)
def decision(req: PredictRequest):
    preds = predict_churn()
    row = preds.iloc[0]

    explanations = explain_predictions(top_k=5)
    drivers = explanations[0]["top_drivers"]

    decision = choose_best_action(
        customer_id=row["customer_id"],
        churn_probability=row["churn_probability"],
        customer_value=1200,  # placeholder
        uplift_estimates={
            "DISCOUNT_10": 0.18,
            "PRIORITY_SUPPORT": 0.05,
            "LOYALTY_OFFER": 0.02
        }
    )

    return DecisionResponse(
        request_id=row["request_id"],
        customer_id=row["customer_id"],
        churn_probability=float(row["churn_probability"]),
        model_version=row["model_version"],
        feature_version=row["feature_version"],
        top_drivers=drivers,
        recommended_action=decision["recommended_action"],
        expected_value=decision["expected_value"]
    )