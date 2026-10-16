from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def test_dataset_schema_types_and_missing_values() -> None:
    frame = pd.read_csv(PROJECT_ROOT / "data" / "heart.csv")

    assert list(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 303
    assert frame.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes)
    assert set(frame["target"].unique()) == {0, 1}
