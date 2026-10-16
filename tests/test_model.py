from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import FEATURE_COLUMNS, HeartDiseasePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_preprocessor_shape_and_probability_bounds(tmp_path: Path) -> None:
    frame = pd.read_csv(PROJECT_ROOT / "data" / "heart.csv")
    features = frame[FEATURE_COLUMNS]
    labels = frame["target"]

    preprocessor = HeartDiseasePreprocessor().fit(features)
    transformed = preprocessor.transform(features)
    assert transformed.shape == features.shape
    assert transformed.isna().sum().sum() == 0

    integer_features = features.iloc[[0]].astype(
        {column: "int64" for column in ["age", "trestbps", "chol", "thalach"]}
    )
    integer_features["oldpeak"] = integer_features["oldpeak"].astype("int64")
    assert (
        preprocessor.transform(integer_features).shape
        == integer_features.shape
    )

    saved_preprocessor = tmp_path / "preprocessor.pkl"
    preprocessor.save(saved_preprocessor)
    restored = HeartDiseasePreprocessor.load(saved_preprocessor)
    pd.testing.assert_frame_equal(transformed, restored.transform(features))

    pipeline = Pipeline(
        [
            ("preprocessor", HeartDiseasePreprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    ).fit(features, labels)
    probabilities = pipeline.predict_proba(features)
    assert probabilities.shape == (len(frame), 2)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()

    saved_model = tmp_path / "model.pkl"
    joblib.dump(pipeline, saved_model)
    assert joblib.load(saved_model).predict_proba(features[:2]).shape == (2, 2)
