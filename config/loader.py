import yaml
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"


def load_config():
    """Load configuration from config.yaml"""
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


def get_policy_config():
    """Get policy evaluation configuration"""
    config = load_config()
    return config.get("policy", {})


def get_ui_config():
    """Get UI configuration"""
    config = load_config()
    return config.get("ui", {})


def get_actions_config():
    """Get actions configuration"""
    config = load_config()
    return config.get("actions", {})


def get_models_config():
    """Get models configuration"""
    config = load_config()
    return config.get("models", {})


def get_customer_value():
    """Get customer lifetime value"""
    return get_policy_config().get("customer_value", 1200)


def get_default_churn_probability():
    """Get default churn probability fallback"""
    return get_policy_config().get("default_churn_probability", 0.5)


def get_risk_thresholds():
    """Get risk threshold configuration"""
    return get_ui_config().get("risk_thresholds", {
        "high_risk": 0.6,
        "medium_risk": 0.3,
        "low_risk": 0.0
    })


def get_action_cost(action_name: str) -> float:
    """Get cost for a specific action"""
    actions = get_actions_config()
    return actions.get(action_name, {}).get("cost", 0)
