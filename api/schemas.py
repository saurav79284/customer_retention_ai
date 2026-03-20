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


# =====================================================
# A/B Test Simulator Schemas
# =====================================================

class SimulateRequest(BaseModel):
    """Request schema for A/B test simulation."""
    n_customers: int = 1000
    customer_value: float = 1200
    uplift_discount_10: float = 0.18
    uplift_priority_support: float = 0.05
    uplift_loyalty_offer: float = 0.02


class SimulateMetricsResponse(BaseModel):
    """Response schema for A/B test metrics."""
    ai_total_revenue: float
    baseline_total_revenue: float
    ai_total_cost: float
    ai_net_revenue: float
    baseline_net_revenue: float
    roi_percentage: float
    revenue_lift_percentage: float
    retention_rate_ai: float
    retention_rate_baseline: float
    avg_eu_per_customer_ai: float
    avg_eu_per_customer_baseline: float
    p_value: float
    statistically_significant: bool
    sample_size: int
    t_statistic: float


class ActionDistributionResponse(BaseModel):
    """Response schema for action distribution."""
    NO_ACTION: int
    DISCOUNT_10: int
    PRIORITY_SUPPORT: int
    LOYALTY_OFFER: int


class SimulateResponse(BaseModel):
    """Response schema for A/B test simulation."""
    simulation_id: str
    metrics: SimulateMetricsResponse
    action_distribution: ActionDistributionResponse
    campaign_summary: Dict[str, Any]