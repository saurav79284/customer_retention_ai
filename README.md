# 🎯 Customer Retention AI: Intelligent Churn Prediction & Policy Evaluation System

An advanced machine learning platform that predicts customer churn, recommends personalized retention actions, and evaluates campaign effectiveness through statistically rigorous A/B testing simulations.

**Live Demo**: [Streamlit Cloud App](https://customerretentionai.streamlit.app)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Dashboard Features](#dashboard-features)
- [How It Works](#how-it-works)
- [Deployment](#deployment)
- [Testing](#testing)
- [Contributing](#contributing)

---

## 🚀 Overview

This project leverages machine learning and causal inference to predict customer churn and optimize retention strategies. It combines:

- **Predictive Analytics**: LightGBM models for accurate churn probability estimation
- **Causal Inference**: T-Learner for heterogeneous treatment effect estimation across different actions
- **Decision Optimization**: Expected Value (EV) formula to recommend personalized actions
- **Policy Evaluation**: Compare AI-recommended policies against baseline strategies
- **Statistical Testing**: Paired t-tests for A/B test significance validation
- **Explainability**: SHAP values for interpretable predictions

---

## ✨ Key Features

### 🤖 Machine Learning

- **Churn Prediction**: LightGBM classifier on 7,043 customer dataset
- **Propensity Scoring**: Estimate probability of action response
- **Treatment Effects**: T-Learner models for four retention actions
- **Feature Engineering**: Automated feature pipeline with SQL integration

### 💡 Decision Engine

- **Personalized Actions**: Recommends optimal action per customer
- **Expected Value Calculation**: `EV = (Uplift × Customer Value) - Action Cost`
- **Four Action Types**:
  - 🎁 DISCOUNT_10: 10% discount (12% uplift, ₹50 cost)
  - 🎧 PRIORITY_SUPPORT: Priority tier (11% uplift, ₹30 cost)
  - ⭐ LOYALTY_OFFER: Loyalty program (16% uplift, ₹20 cost)
  - ✋ NO_ACTION: Baseline control

### 📊 Policy Evaluation

- **Dynamic Evaluation**: Real-time policy comparison
- **Smart Sampling**: Optimized 1000-customer sampling (5-15 seconds)
- **Metrics**: Revenue lift, ROI, retention rates
- **Statistical Significance**: Paired t-test with configurable thresholds

### 🧪 A/B Test Simulator

- **Campaign Simulation**: End-to-end A/B test simulation
- **Realistic Outcomes**: Uses actual churn predictions per customer
- **Uplift Adjustment**: Configurable uplift via interactive sliders
- **Comprehensive Metrics**:
  - Revenue analysis (gross, net, per-customer)
  - Retention rates (AI vs baseline)
  - ROI percentage
  - Statistical significance (p-value)

### 🎨 Interactive Dashboard

- **Streamlit Web UI**: Real-time policy evaluation and exploration
- **SHAP Explainability**: Feature importance for individual predictions
- **Dynamic Filtering**: Risk-based customer segmentation
- **Campaign Metrics**: Visual analytics for decision makers

---

## 🛠 Tech Stack

### Core ML & Data

| Component           | Library      | Version |
| ------------------- | ------------ | ------- |
| ML Framework        | scikit-learn | ≥1.4.0  |
| Gradient Boosting   | LightGBM     | ≥4.0.0  |
| Data Processing     | Pandas       | ≥2.0.0  |
| Numerical Computing | NumPy        | ≥1.24.0 |
| Statistics          | SciPy        | ≥1.10.0 |

### Explainability & Visualization

| Component          | Library   | Version |
| ------------------ | --------- | ------- |
| SHAP Explanations  | SHAP      | ≥0.42.0 |
| Interactive Charts | Plotly    | ≥5.14.0 |
| Web Framework      | Streamlit | ≥1.28.0 |

### Utilities

| Component     | Library | Version |
| ------------- | ------- | ------- |
| Configuration | PyYAML  | ≥6.0    |
| Serialization | joblib  | ≥1.3.0  |
| Images        | Pillow  | ≥10.0.0 |

---

## 🏗 Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                      │
│  (Interactive UI for exploration & decision making)        │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼──────┐ ┌──▼─────┐ ┌───▼──────┐
   │   Policy  │ │  A/B   │ │ Customer │
   │ Evaluation│ │ Sim    │ │ Explorer │
   └────┬──────┘ └──┬─────┘ └───┬──────┘
        │           │            │
        └───────────┼────────────┘
                    │
        ┌───────────▼────────────┐
        │  DECISION ENGINE       │
        │  ┌─────────────────┐   │
        │  │ Action Selection│   │
        │  │ (EV Formula)    │   │
        │  └────────┬────────┘   │
        └───────────┼────────────┘
                    │
        ┌───────────▼────────────┐
        │  EVALUATION MODELS     │
        │ ┌──────┐ ┌──────────┐  │
        │ │Churn │ │Propensity│  │
        │ │Model │ │ Models   │  │
        │ └──────┘ └──────────┘  │
        │ ┌──────────────────┐   │
        │ │ T-Learner Models │   │
        │ │ (4 actions)      │   │
        │ └──────────────────┘   │
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │   DATA LAYER           │
        │ ┌──────────────────┐   │
        │ │ Customer Data    │   │
        │ │ (7,043 customers)│   │
        │ └──────────────────┘   │
        └────────────────────────┘
```

### Component Breakdown

**decision_engine/**

- `actions.py`: Reward calculation & outcome simulation
- `expected_value.py`: EV formula & action selection
- `impact_model.py`: Treatment effect estimation
- `policy.py`: Policy evaluation & comparison
- `t_learner.py`: T-Learner model training
- `uplift_provider.py`: Uplift effect management
- `ab_test_simulator.py`: Campaign simulation engine

**models/**

- `train_lightgbm.py`: Churn prediction model training
- `predict.py`: Model inference pipeline
- `artifacts/`: Serialized models & metadata

**evaluation/**

- `propensity_model.py`: Action response probability
- `outcome_model.py`: Outcome prediction
- `ips_estimator.py`: Inverse probability scoring
- `doubly_robust.py`: DR estimation for treatment effects
- `policy_evaluation.py`: Policy comparison & metrics

**ui/**

- `streamlit_app.py`: Interactive dashboard

**config/**

- `config.yaml`: Parameterized configuration
- `loader.py`: Configuration management

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- pip or conda
- Git

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit app
cd customer_retention_ai
streamlit run ui/streamlit_app.py
```

The app will open at `http://localhost:8506`

---

## 🚀 Quick Start

### Running Locally

```bash
# From project root
cd customer_retention_ai
streamlit run ui/streamlit_app.py
```

### Using the Dashboard

1. **Churn Risk Assessment**
   - Select a customer ID
   - View predicted churn probability
   - See SHAP feature importance

2. **Recommended Actions**
   - View AI-recommended action
   - See Expected Value breakdown
   - Compare action costs vs uplift

3. **Policy Evaluation**
   - Click "Refresh Policy" button
   - Compare AI policy vs baseline
   - View improvement metrics

4. **A/B Test Simulator**
   - Adjust campaign size and uplift estimates
   - Click "Run Simulation"
   - View revenue, ROI, and retention metrics

---

## 📁 Project Structure

```
customer_retention_ai/
│
├── config/
│   ├── config.yaml                 # Configuration (customer value, costs, thresholds)
│   └── loader.py                   # Config management utilities
│
├── decision_engine/
│   ├── actions.py                  # Action rewards & outcome simulation
│   ├── expected_value.py           # EV formula & action selection
│   ├── impact_model.py             # Treatment effect modeling
│   ├── policy.py                   # Policy evaluation & comparison
│   ├── t_learner.py                # T-Learner implementation
│   ├── uplift_provider.py          # Uplift management
│   └── ab_test_simulator.py        # A/B test simulation engine
│
├── models/
│   ├── train_lightgbm.py           # Churn model training
│   ├── predict.py                  # Inference pipeline
│   ├── feature_schema.json         # Feature definitions
│   └── artifacts/                  # Serialized models
│       ├── churn_model.pkl
│       ├── propensity_models/
│       ├── t_learner_models/
│       └── model_metadata.json
│
├── evaluation/
│   ├── propensity_model.py         # Action response probability
│   ├── outcome_model.py            # Customer retention outcome
│   ├── ips_estimator.py            # Inverse probability scoring
│   ├── doubly_robust.py            # DR estimation
│   └── policy_evaluation.py        # Policy metrics
│
├── features/
│   ├── build_features.py           # Feature engineering
│   └── sql/                        # SQL feature queries
│
├── ui/
│   └── streamlit_app.py            # Interactive dashboard
│
├── explainability/
│   └── shap_explainer.py           # SHAP explanations
│
├── monitoring/
│   └── drift_monitor.py            # Data drift detection
│
├── training/
│   ├── load_data.py                # Data loading
│   ├── train_propensity_model.py    # Propensity training
│   ├── train_t_learner.py          # T-Learner training
│   └── simulate_actions.py         # Action simulation
│
├── tests/
│   ├── test_*.py                   # Unit tests
│   └── README.md                   # Test documentation
│
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv
│
├── requirements.txt                # Python dependencies
├── LICENSE
└── README.md
```

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize behavior:

```yaml
# Policy Settings
policy:
  customer_value: 1200 # Revenue per retained customer (₹)
  default_churn_probability: 0.5 # Reference churn rate

# Risk Thresholds (UI)
risk_thresholds:
  high: 0.6 # Churn probability > 60%
  medium: 0.3 # 30% - 60%

# Action Costs (₹)
actions:
  DISCOUNT_10:
    cost: 50.0
    name: "10% Discount"
  PRIORITY_SUPPORT:
    cost: 30.0
    name: "Priority Support"
  LOYALTY_OFFER:
    cost: 20.0
    name: "Loyalty Offer"
  NO_ACTION:
    cost: 0.0
    name: "No Action (Baseline)"

# Default Uplift Estimates (% churn reduction)
uplift_estimates:
  DISCOUNT_10: 0.12 # 12% effectiveness
  PRIORITY_SUPPORT: 0.11 # 11% effectiveness
  LOYALTY_OFFER: 0.16 # 16% effectiveness (most effective)
  NO_ACTION: 0.0
```

---

## 📊 Dashboard Features

### 1. **Churn Risk Prediction**

- Individual customer risk scores
- SHAP feature importance
- Risk segmentation (High/Medium/Low)

### 2. **Explainability**

- Feature impact visualization
- SHAP force plots
- Contribution breakdown

### 3. **Recommended Actions**

- Optimal action per customer
- Expected value calculation
- Risk-adjusted recommendations

### 4. **Policy Evaluation**

- Dynamic policy comparison
- AI vs baseline metrics
- Improvement percentage
- Real-time refresh

### 5. **A/B Test Simulator**

- Campaign size configuration
- Uplift adjustment sliders
- Revenue & ROI metrics
- Retention rate analysis
- Per-customer economics

---

## 🔍 How It Works

### Step 1: Churn Prediction

```
Customer Features → LightGBM Model → Churn Probability
```

### Step 2: Action Selection

```
For Each Customer:
  ├─ Estimate propensity to respond
  ├─ Estimate treatment effects (T-Learner)
  ├─ Calculate EV for each action
  │   EV = (Uplift × Customer Value) - Action Cost
  └─ Select action with MAX(EV)
```

### Step 3: Policy Evaluation

```
Sample 1000 Customers:
  ├─ Get AI recommendations (Step 2)
  ├─ Simulate outcomes with uplift
  ├─ Calculate metrics:
  │  ├─ Total revenue (AI vs baseline)
  │  ├─ Campaign costs
  │  ├─ Net revenue & ROI
  │  ├─ Retention rates
  │  └─ Statistical significance (paired t-test)
  └─ Compare AI policy vs NO_ACTION baseline
```

### Step 4: A/B Test Simulation

```
For Each Customer:
  ├─ Get churn probability
  ├─ Apply AI action
  ├─ Simulate outcome with uplift
  ├─ Calculate reward & net benefit
  └─ Aggregate metrics
```

---

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. **Push to GitHub**

   ```bash
   git init
   git add .
   git commit -m "Initial commit: Customer retention AI"
   git remote add origin https://github.com/YOUR_USERNAME/customer-churn-prediction.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [https://share.streamlit.io](https://share.streamlit.io)
   - Sign up with GitHub
   - Click "New app"
   - Select repo, branch: `main`
   - Main file: `customer_retention_ai/ui/streamlit_app.py`
   - Click Deploy

3. **Share URL**
   - Streamlit generates: `https://customer-churn-prediction-XXXXX.streamlit.app`

### Environment Variables

If deploying with sensitive data:

1. Go to app **Settings** (⚙️)
2. Add secrets in format:
   ```
   DATABASE_URL = "postgresql://..."
   API_KEY = "..."
   ```

---

## 🧪 Testing

Run all tests:

```bash
cd customer_retention_ai
python -m pytest tests/ -v
```

Run specific test:

```bash
python -m pytest tests/test_policy.py -v
```

Test coverage:

```bash
pytest --cov=customer_retention_ai tests/
```

### Test Modules

- `test_policy.py`: Policy evaluation logic
- `test_expected_value.py`: EV calculation
- `test_outcome_model.py`: Outcome prediction
- `test_propensity_model.py`: Propensity scoring
- `test_doubly_robust.py`: DR estimation
- `test_features.py`: Feature engineering
- `test_inference.py`: Model inference
- `test_shap.py`: SHAP explanations

---

## 📊 Key Metrics

### Policy Evaluation

- **Revenue Lift %**: Improvement from AI policy
- **ROI %**: Return on investment
- **Retention Rate**: % of customers retained
- **Net Revenue**: Revenue minus campaign costs

### A/B Test Simulator

- **Campaign Cost**: Total spending on actions
- **Revenue Lift**: Additional revenue from campaigns
- **Per-Customer Economics**: Average EU improvement
- **Statistical Significance**: p-value < 0.05 = significant

---

## 🔐 Data Privacy

- Models trained on 7,043 anonymized customers
- No personal identifying information (PII) in features
- Feature schema in `models/feature_schema.json`
- All data handling complies with privacy best practices

---

## 📚 References

### Papers & Methods

- **T-Learner**: Kunzel et al., 2019 - "Metalearners for Estimating HTE"
- **Doubly Robust**: Kennedy, 2020 - "Optimal Doubly Robust Estimation"
- **SHAP**: Lundberg & Lee, 2017 - "A Unified Approach to Interpreting Model Predictions"

### Libraries

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👥 Author

**Saurav Raj** - [GitHub Profile](https://github.com/saurav79284)

---

## 📞 Support

For issues, questions, or suggestions:

- Open an [Issue](https://github.com/saurav79284/customer-churn-prediction/issues)
- Check existing [Discussions](https://github.com/saurav79284/customer-churn-prediction/discussions)

---

## 🎉 Acknowledgments

- Kaggle Telco Customer Churn Dataset
- Streamlit team for amazing web framework
- Open source ML community

---

**Last Updated**: March 2026
