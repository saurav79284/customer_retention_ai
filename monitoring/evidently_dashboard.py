import pandas as pd
from pathlib import Path

from evidently import Dataset, Report, ColumnMapping
from evidently.presets import DataDriftPreset, DataSummaryPreset

REFERENCE_DATA_PATH = Path("data/processed/features.parquet")
CURRENT_DATA_PATH = Path("data/processed/features.parquet")
REPORT_PATH = Path("monitoring/evidently_report.html")


def load_data():
    reference = pd.read_parquet(REFERENCE_DATA_PATH)
    current = pd.read_parquet(CURRENT_DATA_PATH)

    drop_cols = ["customer_id", "feature_version"]
    reference = reference.drop(columns=drop_cols, errors="ignore")
    current = current.drop(columns=drop_cols, errors="ignore")

    return reference, current


def generate_report():
    reference_df, current_df = load_data()

    column_mapping = ColumnMapping(
        target=None,
        prediction=None,
        numerical_features=reference_df.columns.tolist(),
        categorical_features=[]
    )

    reference = Dataset(reference_df, column_mapping)
    current = Dataset(current_df, column_mapping)

    report = Report(
        presets=[
            DataSummaryPreset(),
            DataDriftPreset(),
        ]
    )

    report.run(reference_data=reference, current_data=current)
    report.save_html(REPORT_PATH)

    print(f"Evidently report saved to: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    generate_report()