# 🎯 Customer Retention AI: Churn Prediction & Policy Evaluation

ML-powered platform that predicts customer churn, recommends personalized retention actions, and evaluates campaign effectiveness through A/B testing.

**Live Demo**: [Streamlit Cloud](https://customerretentionai.streamlit.app)

---

## ✨ Features

- **Churn Prediction**: LightGBM model on 7,043 customer dataset
- **Decision Engine**: Expected Value optimization recommending best action per customer
- **Policy Evaluation**: Real-time comparison of AI policy vs baseline (no action)
- **A/B Simulator**: Campaign simulation with ROI, revenue, and retention metrics
- **SHAP Explainability**: Feature importance for individual predictions
- **Streamlit Dashboard**: Interactive UI for exploration and decision-making

---

## 🛠 Tech Stack

| Component      | Library                | Version |
| -------------- | ---------------------- | ------- |
| ML Framework   | scikit-learn, LightGBM | ≥1.4.0  |
| Data           | Pandas, NumPy          | ≥2.0.0  |
| Statistics     | SciPy                  | ≥1.10.0 |
| Explainability | SHAP                   | ≥0.42.0 |
| Visualization  | Plotly                 | ≥5.14.0 |
| Web Framework  | Streamlit              | ≥1.28.0 |
| Config         | PyYAML                 | ≥6.0    |

---

## 🏗 Architecture

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

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```bash
cd customer_retention_ai
streamlit run ui/streamlit_app.py
```

Open `http://localhost:8506`

### Dashboard Usage

1. **Churn Risk**: Select customer → view risk score & SHAP explanations
2. **Actions**: See AI recommendation & Expected Value breakdown
3. **Policy**: Click "Refresh Policy" → compare AI vs baseline metrics
4. **A/B Test**: Adjust sliders → run simulation → view revenue & ROI

---

## 📁 Project Structure

```
customer_retention_ai/
├── decision_engine/       # Action selection & policy evaluation
├── models/                # Trained ML models & inference
├── evaluation/            # Outcome prediction & causal effects
├── features/              # Feature engineering pipeline
├── ui/                    # Streamlit dashboard
├── config/                # Configuration (config.yaml, loader.py)
├── tests/                 # Unit tests
└── data/raw/              # Customer churn dataset
```

## 📊 How It Works

```
1. Customer Features → LightGBM → Churn Probability
2. For Each Customer:
   - Estimate propensity & treatment effects
   - Calculate EV = (Uplift × Customer Value) - Cost
   - Select action with MAX(EV)
3. Policy Evaluation: Compare AI vs baseline on 1000 customers
4. Metrics: Revenue lift, ROI, retention rate, statistical significance (p-value)
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

**Last Updated**: March 2026
