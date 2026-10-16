import numpy as np
from fastapi.testclient import TestClient

from api.main import create_app


class StubModel:
    classes_ = np.array([0, 1])

    def predict(self, features: object) -> np.ndarray:
        return np.array([1])

    def predict_proba(self, features: object) -> np.ndarray:
        return np.array([[0.2, 0.8]])


PATIENT = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


def test_root_redirects_to_swagger_ui() -> None:
    client = TestClient(create_app(model=StubModel()))

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health_and_prediction() -> None:
    client = TestClient(create_app(model=StubModel()))

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }

    prediction_response = client.post("/predict", json=PATIENT)
    assert prediction_response.status_code == 200
    assert prediction_response.json() == {"prediction": 1, "confidence": 0.8}


def test_invalid_prediction_input_returns_422() -> None:
    client = TestClient(create_app(model=StubModel()))
    invalid_patient = {**PATIENT, "sex": 2}

    response = client.post("/predict", json=invalid_patient)
    assert response.status_code == 422


def test_metrics_exposes_request_and_prediction_counters() -> None:
    client = TestClient(create_app(model=StubModel()))
    client.post("/predict", json=PATIENT)

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert "predictions_total" in response.text
