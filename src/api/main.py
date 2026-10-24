"""FastAPI service for HAI anomaly scoring and simulated risk assessment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tensorflow import keras

from src.api.database import EventStore
from src.application.detection_service import DetectionService
from src.infrastructure.config import AppConfig
from src.zero_trust.risk_engine import RiskInput, calculate_risk


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "autoencoder.keras"
THRESHOLD_PATH = ROOT / "models" / "anomaly_threshold.json"
METRICS_PATH = ROOT / "reports" / "baseline_metrics.json"
DATASET_PROFILE_PATH = ROOT / "reports" / "dataset-profile.json"
DATABASE_PATH = ROOT / "data" / "resinet-events.sqlite3"


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    identity_trust: float = Field(ge=0.0, le=1.0)
    device_trust: float = Field(ge=0.0, le=1.0)
    asset_criticality: float = Field(ge=0.0, le=1.0)
    behavior_deviation: float = Field(ge=0.0, le=1.0)

    @field_validator(
        "identity_trust",
        "device_trust",
        "asset_criticality",
        "behavior_deviation",
    )
    @classmethod
    def reject_non_finite(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("context values must be finite")
        return value


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    features: dict[str, float]
    context: ContextRequest
    review_status: Literal["NEW", "IN_REVIEW", "RESOLVED"] = "NEW"

    @field_validator("features")
    @classmethod
    def reject_non_finite_features(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("features must not be empty")
        if not all(np.isfinite(number) for number in value.values()):
            raise ValueError("features must contain only finite numbers")
        return value


class EventResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_id: str
    timestamp: str
    asset_id: str
    anomaly_score: float
    threshold: float
    anomaly_decision: bool
    risk_score: float
    risk_level: str
    review_status: str
    identity_id: str
    device_id: str
    context: dict[str, Any]
    risk_components: dict[str, float]


def _load_metadata() -> dict[str, Any]:
    if not THRESHOLD_PATH.exists():
        raise RuntimeError(f"Threshold artifact not found: {THRESHOLD_PATH}")
    return json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))


class ScoringService:
    def __init__(self) -> None:
        self.metadata = _load_metadata()
        config = AppConfig.from_environment(ROOT)
        self.detection = DetectionService(ROOT, config.database_path)

    @property
    def store(self) -> EventStore:
        return self.detection.store

    @store.setter
    def store(self, value: EventStore) -> None:
        self.detection.store = value

    def score(self, request: ScoreRequest) -> dict[str, Any]:
        return self.detection.detect(
            "OT",
            request.features,
            timestamp=request.timestamp,
            identity_id=request.context.identity_id,
            device_id=request.context.device_id,
            asset_id=request.context.asset_id,
            identity_trust=request.context.identity_trust,
            device_trust=request.context.device_trust,
            asset_criticality=request.context.asset_criticality,
            behavior_deviation=request.context.behavior_deviation,
            review_status=request.review_status,
        )


app = FastAPI(title="ResiNet-IDAM API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
service = ScoringService()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    return service.store.summary()


@app.get("/api/events", response_model=list[EventResponse])
def events(
    risk_level: str | None = Query(default=None),
    asset_id: str | None = Query(default=None),
    review_status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    return service.store.list(risk_level, asset_id, review_status, limit)


@app.get("/api/events/{event_id}", response_model=EventResponse)
def event(event_id: str) -> dict[str, Any]:
    result = service.store.get(event_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return result


@app.get("/api/model/info")
def model_info() -> dict[str, Any]:
    return {
        "model_version": service.metadata.get("model_version"),
        "threshold": service.metadata["threshold"],
        "threshold_selection": service.metadata["threshold_selection"],
        "feature_count": len(service.metadata["feature_columns"]),
        "feature_columns": service.metadata["feature_columns"],
    }


@app.get("/api/model/metrics")
def model_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="Model metrics are unavailable")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.get("/api/dataset/profile")
def dataset_profile() -> dict[str, Any]:
    if not DATASET_PROFILE_PATH.exists():
        raise HTTPException(status_code=503, detail="Dataset profile is unavailable")
    return json.loads(DATASET_PROFILE_PATH.read_text(encoding="utf-8"))


@app.post("/api/events/score", response_model=EventResponse, status_code=201)
def score_event(request: ScoreRequest) -> dict[str, Any]:
    try:
        return service.score(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
