from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class PredictRequest(BaseModel):
    customer_id: Optional[str] = None


class PredictResponse(BaseModel):
    request_id: str
    customer_id: str
    churn_probability: float
    model_version: str
    feature_version: str


class DecisionResponse(PredictResponse):
    top_drivers: List[Dict[str, Any]]
    recommended_action: str
    expected_value: float