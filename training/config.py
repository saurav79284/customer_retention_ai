# customer_retention_ai/training/config.py

# Use all numeric features except labels
EXCLUDED_COLUMNS = {
    "customer_id",
    "churn",
    "action"   # added later by simulation
}

def get_feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDED_COLUMNS]