"""Reusable imputation and scaling for heart disease features."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
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
]
CONTINUOUS_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in CONTINUOUS_COLUMNS
]


class HeartDiseasePreprocessor(BaseEstimator, TransformerMixin):
    """Impute feature values and standardize continuous columns."""

    def __init__(self) -> None:
        self.medians_: dict[str, float] = {}
        self.modes_: dict[str, float] = {}
        self.scaler_: StandardScaler | None = None

    def _validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "HeartDiseasePreprocessor expects a pandas DataFrame"
            )
        missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        return frame.loc[:, FEATURE_COLUMNS].copy()

    def fit(
        self, frame: pd.DataFrame, y: object = None
    ) -> HeartDiseasePreprocessor:
        features = self._validate_frame(frame)
        for column in CONTINUOUS_COLUMNS:
            values = pd.to_numeric(features[column], errors="raise")
            if values.notna().sum() == 0:
                raise ValueError(
                    f"Cannot impute column with no observed values: {column}"
                )
            self.medians_[column] = float(values.median())
            features[column] = values.fillna(self.medians_[column])

        for column in CATEGORICAL_COLUMNS:
            values = pd.to_numeric(features[column], errors="raise")
            modes = values.mode(dropna=True)
            if modes.empty:
                raise ValueError(
                    f"Cannot impute column with no observed values: {column}"
                )
            self.modes_[column] = float(modes.iloc[0])
            features[column] = values.fillna(self.modes_[column])

        self.scaler_ = StandardScaler()
        self.scaler_.fit(features[CONTINUOUS_COLUMNS])
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        features = self._validate_frame(frame)
        for column in CONTINUOUS_COLUMNS:
            features[column] = pd.to_numeric(
                features[column], errors="raise"
            ).fillna(self.medians_[column])
        for column in CATEGORICAL_COLUMNS:
            features[column] = pd.to_numeric(
                features[column], errors="raise"
            ).fillna(self.modes_[column])

        scaled = self.scaler_.transform(features[CONTINUOUS_COLUMNS])
        for index, column in enumerate(CONTINUOUS_COLUMNS):
            features[column] = scaled[:, index]
        return features

    def save(self, path: str | Path) -> Path:
        """Serialize the fitted preprocessor to disk."""
        if self.scaler_ is None:
            raise RuntimeError("Cannot serialize an unfitted preprocessor")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> HeartDiseasePreprocessor:
        """Load a serialized preprocessor and validate its type."""
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Expected a serialized {cls.__name__}")
        return instance
