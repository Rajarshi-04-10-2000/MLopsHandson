"""Download and prepare the UCI Cleveland Heart Disease dataset."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
DATA_COLUMNS = [
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
CONTINUOUS_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "heart.csv"


def load_cleveland_data() -> pd.DataFrame:
    """Fetch the archive and return a clean binary-label dataframe."""
    with urlopen(DATA_URL, timeout=30) as response:
        archive_bytes = response.read()

    with ZipFile(BytesIO(archive_bytes)) as archive:
        source_name = next(
            (
                name
                for name in archive.namelist()
                if name.endswith("processed.cleveland.data")
            ),
            None,
        )
        if source_name is None:
            raise ValueError("UCI archive does not contain Cleveland data")
        with archive.open(source_name) as source:
            frame = pd.read_csv(
                source,
                header=None,
                names=DATA_COLUMNS,
                na_values="?",
                skipinitialspace=True,
            )

    if len(frame) != 303:
        raise ValueError(f"Expected 303 Cleveland records, found {len(frame)}")

    for column in CONTINUOUS_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        frame[column] = frame[column].fillna(frame[column].median())
    for column in CATEGORICAL_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        modes = frame[column].mode(dropna=True)
        if modes.empty:
            raise ValueError(
                f"Cannot impute column with no observed values: {column}"
            )
        frame[column] = frame[column].fillna(modes.iloc[0]).astype("int64")

    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(
        "int64"
    )
    frame["target"] = (frame["target"] > 0).astype("int64")
    frame = frame[DATA_COLUMNS]

    if frame.isna().any().any():
        raise ValueError("Prepared dataset still contains missing values")
    return frame


def download_data(output_path: Path = OUTPUT_PATH) -> Path:
    """Download, validate, and write the processed CSV."""
    frame = load_cleveland_data()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Saved {len(frame)} rows to {output_path}")
    return output_path


if __name__ == "__main__":
    download_data()
