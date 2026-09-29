import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR.parent / "data" / "raw" / "dataset.csv"

df = pd.read_csv(DATASET, encoding="utf-8")

EXPECTED_COLUMNS = [
    "CUST_ID",
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
]

NUMERIC_COLUMNS = [
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
]


def validate_schema(df: pd.DataFrame) -> bool:
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        return False
    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            return False
    return True


if __name__ == "__main__":
    validate_schema(df)
