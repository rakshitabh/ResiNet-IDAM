"""Execution result model for simulated events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SimulationEventResult:
    event_id: str
    source: str
    event_type: str
    user_id: str
    device_id: str
    asset_id: str
    scheduled_offset: float
    scheduled_time_iso: str
    actual_dispatch_time_iso: str
    completion_time_iso: str
    dispatch_jitter_ms: float
    processing_latency_ms: float
    detector_prediction: str
    anomaly_score: float
    detection_threshold: float
    is_anomaly: bool
    risk_score: float
    risk_level: str
    policy_verdict: str
    recommended_action: str
    policy_reason: str
    top_contributing_features: list[dict[str, Any]] = field(default_factory=list)
    rule_traces: list[dict[str, Any]] = field(default_factory=list)
    risk_components: dict[str, float] = field(default_factory=dict)
    expected_outcome: str | None = None
    outcome_matched: bool | None = None
    ground_truth_label: str | None = None
    is_attack_ground_truth: bool | None = None
    classification_result: str = "UNAVAILABLE"
    error_message: str | None = None
    status: str = "COMPLETED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "asset_id": self.asset_id,
            "scheduled_offset": self.scheduled_offset,
            "scheduled_time_iso": self.scheduled_time_iso,
            "actual_dispatch_time_iso": self.actual_dispatch_time_iso,
            "completion_time_iso": self.completion_time_iso,
            "dispatch_jitter_ms": round(self.dispatch_jitter_ms, 3),
            "processing_latency_ms": round(self.processing_latency_ms, 3),
            "detector_prediction": self.detector_prediction,
            "anomaly_score": round(self.anomaly_score, 6),
            "detection_threshold": round(self.detection_threshold, 6),
            "is_anomaly": self.is_anomaly,
            "risk_score": round(self.risk_score, 2),
            "risk_level": self.risk_level,
            "policy_verdict": self.policy_verdict,
            "recommended_action": self.recommended_action,
            "policy_reason": self.policy_reason,
            "top_contributing_features": self.top_contributing_features,
            "rule_traces": self.rule_traces,
            "risk_components": self.risk_components,
            "expected_outcome": self.expected_outcome,
            "outcome_matched": self.outcome_matched,
            "ground_truth_label": self.ground_truth_label,
            "is_attack_ground_truth": self.is_attack_ground_truth,
            "classification_result": self.classification_result,
            "error_message": self.error_message,
            "status": self.status,
        }

