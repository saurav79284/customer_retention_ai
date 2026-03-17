# Customer Retention AI

An end-to-end example of churn risk prediction, uplift modeling, policy optimization, monitoring, and a Streamlit front-end for customer success teams.

## What this project shows
- Predict churn propensity and treatment uplift (which offer most reduces churn for a customer).
- Cost/LTV-aware policy logic that can recommend discounts, loyalty offers, or no action.
- FastAPI microservice for scoring + policy decisions.
- Streamlit UI for CX/CS teams with business-friendly explanations and a technical view.
- Monitoring with Evidently for data/score drift.

## Architecture (high level)
1. **Training** (`training/`): build features, simulate actions, train propensity + uplift/T-learner models.
2. **Models** (`models/`): packaged artifacts + schema + inference code (`predict.py`).
3. **Decision engine** (`decision_engine/`): policy & expected value logic using model scores and action costs.
4. **API** (`api/main.py`): FastAPI endpoints for scoring + action recommendation.
5. **UI** (`ui/streamlit_app.py`): Streamlit app for business and technical users.
6. **Monitoring** (`monitoring/`): drift detection and reports.

## Quickstart
```bash
# 1) Install (Python 3.9+ recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Run API
tuvicorn api.main:app --reload --port 8000
# Test
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d @sample_request.json

# 3) Run Streamlit UI (uses API)
streamlit run ui/streamlit_app.py
```

## Sample request body (to `/score`)
```json
{
  "customer_id": "12345",
  "contract": "Month-to-month",
  "tenure": 5,
  "monthly_charges": 78.5,
  "total_charges": 320.0,
  "avg_monthly_spend": 75.0
}
```
Response includes churn_probability, recommended_action, and top_feature_reasons.

## Training workflow
```bash
# Generate/prepare data
python training/load_data.py
# Simulate treatment actions
python training/simulate_actions.py
# Train propensity model
python training/train_propensity_model.py
# Train T-learner uplift
python training/train_t_learner.py
# Train LightGBM single model (baseline)
python models/train_lightgbm.py
```
Artifacts land in `models/` (pkl + feature_schema.json).

## Decision policy
- Uses uplift and cost/LTV assumptions to pick the best action.
- Default fallback: no-action if uplift is low or uncertain.
- Extend `decision_engine/policy.py` to add budget constraints or additional offers.

## Monitoring
- `monitoring/drift_monitor.py` runs data/score drift using Evidently.
- `monitoring/drift_report.csv` shows detected drift; extend to alerting.

## UI notes
- Business/Technical toggle: business view shows plain-language reasons; technical view shows SHAP table.
- See screenshot in repo issues or generate via Streamlit.

## Project structure (selected)
- `api/` – FastAPI app + schemas
- `ui/` – Streamlit UI
- `features/` – feature engineering
- `models/` – training/inference artifacts
- `decision_engine/` – policy + uplift logic
- `monitoring/` – drift checks
- `training/` – data prep & model training scripts
- `tests/` – add smoke tests here (API + predict contract)

## Tech stack
Python, FastAPI, Streamlit, pandas, numpy, scikit-learn/LightGBM, SHAP, Evidently.

## Roadmap
- Add CI (lint + tests)
- Calibrate uplift scores + uncertainty bands
- Budget-aware policy optimization
- Move model artifacts to object storage + manifest
- Add off-policy evaluation for policy changes

## License
MIT
