"""Cross-validate classifiers, track results with MLflow, and export a model.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.preprocessing import FEATURE_COLUMNS, HeartDiseasePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "heart.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_model.pkl"
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def load_training_data(
    path: str | Path = DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the prepared dataset and separate features from binary labels."""
    frame = pd.read_csv(path)
    expected = [*FEATURE_COLUMNS, "target"]
    if list(frame.columns) != expected:
        raise ValueError(
            f"Expected dataset columns {expected}, found {list(frame.columns)}"
        )
    if frame.isna().any().any():
        raise ValueError("Training data contains missing values")
    labels = frame.pop("target").astype("int64")
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Training labels must be binary values 0 and 1")
    return frame, labels


def build_estimators() -> dict[str, Pipeline]:
    """Create pipelines so preprocessing is refit within every CV fold."""
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", HeartDiseasePreprocessor()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=42),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", HeartDiseasePreprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def train_model(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = MODEL_PATH,
) -> tuple[Pipeline, dict[str, dict[str, float]]]:
    """Evaluate both estimators and fit the highest ROC-AUC pipeline."""
    features, labels = load_training_data(data_path)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, dict[str, float]] = {}
    estimators = build_estimators()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-classification")

    for name, estimator in estimators.items():
        with mlflow.start_run(run_name=name):
            scores = cross_validate(
                estimator,
                features,
                labels,
                cv=cv,
                scoring=SCORING,
                n_jobs=1,
                error_score="raise",
            )
            summary = {
                metric: float(scores[f"test_{metric}"].mean())
                for metric in SCORING
            }
            results[name] = summary
            mlflow.log_param("classifier", name)
            mlflow.log_param("cv_splits", cv.n_splits)
            mlflow.log_param("random_state", 42)
            for metric, value in summary.items():
                mlflow.log_metric(f"cv_mean_{metric}", value)
                mlflow.log_metric(
                    f"cv_std_{metric}", float(scores[f"test_{metric}"].std())
                )

    winner = max(results, key=lambda name: results[name]["roc_auc"])
    best_pipeline = estimators[winner].fit(features, labels)
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, destination)
    return best_pipeline, results


if __name__ == "__main__":
    trained_model, evaluation = train_model()
    print(f"Saved winning model to {MODEL_PATH}")
    for name, metrics in evaluation.items():
        print(f"{name}: {metrics}")
