"""Unified security-event contracts shared by detectors, UI, and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

Source = Literal["OT", "IT"]
ReviewStatus = Literal["NEW", "IN_REVIEW", "RESOLVED"]


@dataclass(frozen=True)
class DetectionResult:
    source: Source
    detector: str
    prediction: str
    anomaly_score: float
    threshold: float
    anomaly_decision: bool
    confidence: float | None
    timestamp: datetime
    asset_id: str
    device_id: str
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SecurityEvent:
    event_id: str
    source: Source
    detector: str
    timestamp: str
    asset_id: str
    device_id: str
    identity_id: str
    prediction: str
    anomaly_score: float
    threshold: float
    anomaly_decision: bool
    confidence: float | None
    risk_score: float
    risk_level: str
    review_status: ReviewStatus
    recommended_action: str
    context: dict[str, Any]
    risk_components: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    top_contributing_features: list[dict[str, Any]] = field(default_factory=list)
    fusion_contributions: dict[str, float] = field(default_factory=dict)
    drift_status: str = "NORMAL"
    threshold_mode: str = "ADAPTIVE"
    explanation_summary: str = ""
