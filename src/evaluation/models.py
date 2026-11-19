"""Standardized data models for leakage-safe inference and research evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class ClassificationOutcome(str, Enum):
    TP = "TP"
    FP = "FP"
    TN = "TN"
    FN = "FN"
    UNAVAILABLE = "UNAVAILABLE"  # When ground-truth label is missing or event failed


@dataclass(frozen=True)
class InferencePayload:
    """Payload delivered to detectors during inference.
    
    STRICT SCIENTIFIC CONSTRAINT:
    Contains ONLY feature values and operational routing context.
    Must NEVER contain ground-truth attack labels or expected policy outcomes.
    """
    event_id: str
    source: str  # 'IT' or 'OT'
    features: dict[str, float]
    telemetry_record_id: Optional[str | int] = None
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    asset_id: Optional[str] = None
    timestamp_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRecord:
    """Ground-truth evaluation record kept strictly hidden from detectors until post-inference."""
    event_id: str
    source: str
    ground_truth_label: Optional[str] = None
    is_attack_ground_truth: Optional[bool] = None
    attack_family: Optional[str] = None
    expected_policy_outcome: Optional[str] = None
    is_held_out_unseen: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorOutput:
    """Standardized detector output contract conforming to Phase 3.4."""
    source: str  # 'IT' or 'OT'
    detector_name: str
    prediction: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    confidence: Optional[float] = None
    feature_schema_version: str = "default"
    model_version: str = "v2.1"
    latency_ms: float = 0.0
    top_contributing_features: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "detector_name": self.detector_name,
            "prediction": self.prediction,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": round(self.anomaly_score, 6),
            "threshold": round(self.threshold, 6),
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "feature_schema_version": self.feature_schema_version,
            "model_version": self.model_version,
            "latency_ms": round(self.latency_ms, 3),
            "top_contributing_features": self.top_contributing_features,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationSampleResult:
    """Result of joining model prediction with isolated evaluation ground truth post-inference."""
    event_id: str
    source: str
    prediction: str
    is_predicted_anomaly: bool
    anomaly_score: float
    detection_threshold: float
    classification_outcome: ClassificationOutcome
    ground_truth_label: Optional[str]
    is_attack_ground_truth: Optional[bool]
    is_held_out_unseen: bool
    latency_ms: float
    policy_verdict: Optional[str] = None
    risk_score: Optional[float] = None
    expected_policy_outcome: Optional[str] = None
    outcome_matched: Optional[bool] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source,
            "prediction": self.prediction,
            "is_predicted_anomaly": self.is_predicted_anomaly,
            "anomaly_score": round(self.anomaly_score, 4),
            "detection_threshold": round(self.detection_threshold, 4),
            "classification_outcome": self.classification_outcome.value,
            "ground_truth_label": self.ground_truth_label,
            "is_attack_ground_truth": self.is_attack_ground_truth,
            "is_held_out_unseen": self.is_held_out_unseen,
            "latency_ms": round(self.latency_ms, 3),
            "policy_verdict": self.policy_verdict,
            "risk_score": round(self.risk_score, 2) if self.risk_score is not None else None,
            "expected_policy_outcome": self.expected_policy_outcome,
            "outcome_matched": self.outcome_matched,
            "metadata": self.metadata,
        }

