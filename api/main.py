"""FastAPI inference service with Prometheus request metrics."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_model.pkl"
LOGGER = logging.getLogger("heart_disease_api")
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the API",
    ("method", "path", "status_code"),
)
PREDICTIONS = Counter(
    "predictions_total", "Total successful heart disease predictions"
)


class PatientFeatures(BaseModel):
    """Validated Cleveland dataset features for one patient."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=120)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    trestbps: int = Field(ge=50, le=250)
    chol: int = Field(ge=80, le=700)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int = Field(ge=40, le=250)
    exang: Literal[0, 1]
    oldpeak: float = Field(ge=0, le=10)
    slope: Literal[1, 2, 3]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[3, 6, 7]


class PredictionResponse(BaseModel):
    prediction: int = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


def create_app(
    model: Any | None = None, model_path: str | Path = DEFAULT_MODEL_PATH
) -> FastAPI:
    """Create an application, optionally injecting a model for tests."""
    application = FastAPI(title="Heart Disease Risk API", version="1.0.0")
    application.state.model = model
    application.state.model_path = Path(model_path)

    def resolve_model() -> Any:
        if application.state.model is not None:
            return application.state.model
        path = application.state.model_path
        if not path.is_file():
            raise HTTPException(
                status_code=503, detail="Model artifact is unavailable"
            )
        try:
            loaded_model = joblib.load(path)
        except (OSError, ValueError, EOFError) as error:
            LOGGER.exception("Unable to load model artifact")
            raise HTTPException(
                status_code=503, detail="Model artifact cannot be loaded"
            ) from error
        if not hasattr(loaded_model, "predict") or not hasattr(
            loaded_model, "predict_proba"
        ):
            raise HTTPException(
                status_code=503, detail="Model artifact is invalid"
            )
        application.state.model = loaded_model
        return loaded_model

    @application.middleware("http")
    async def log_request(request: Request, call_next: Any) -> Response:
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            HTTP_REQUESTS.labels(
                request.method, request.url.path, str(status_code)
            ).inc()
            LOGGER.info(
                "method=%s path=%s status_code=%s latency_ms=%.2f",
                request.method,
                request.url.path,
                status_code,
                latency_ms,
            )

    @application.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @application.get("/health")
    def health() -> dict[str, str | bool]:
        try:
            resolve_model()
        except HTTPException as error:
            raise HTTPException(
                status_code=503,
                detail={"status": "unhealthy", "model_loaded": False},
            ) from error
        return {"status": "healthy", "model_loaded": True}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(patient: PatientFeatures) -> PredictionResponse:
        model = resolve_model()
        features = pd.DataFrame([patient.model_dump()])
        prediction = int(model.predict(features)[0])
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 not in classes:
            raise HTTPException(
                status_code=503, detail="Model has no positive class"
            )
        confidence = float(model.predict_proba(features)[0][classes.index(1)])
        PREDICTIONS.inc()
        return PredictionResponse(prediction=prediction, confidence=confidence)

    @application.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()
