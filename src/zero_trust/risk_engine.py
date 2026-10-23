"""Deterministic simulated Zero-Trust risk scoring for anomaly events."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class RiskInput:
    """Simulated context and normalized model signals for one event."""

    identity_trust: float
    device_trust: float
    asset_criticality: float
    anomaly_severity: float
    behavior_deviation: float
    identity_id: str = "simulated-identity"
    device_id: str = "simulated-device"
    asset_id: str = "simulated-asset"


WEIGHTS = {
    "anomaly_severity": 0.40,
    "behavior_deviation": 0.20,
    "identity_untrust": 0.20,
    "device_untrust": 0.10,
    "asset_criticality": 0.10,
}


def _validate_input(data: RiskInput) -> None:
    if not isinstance(data, RiskInput):
        raise TypeError("data must be a RiskInput instance")

    numeric_values = {
        "identity_trust": data.identity_trust,
        "device_trust": data.device_trust,
        "asset_criticality": data.asset_criticality,
        "anomaly_severity": data.anomaly_severity,
        "behavior_deviation": data.behavior_deviation,
    }
    for name, value in numeric_values.items():
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"{name} must be a numeric value between 0.0 and 1.0")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")

    for name in ("identity_id", "device_id", "asset_id"):
        if not isinstance(getattr(data, name), str) or not getattr(data, name).strip():
            raise ValueError(f"{name} must be a non-empty string")


def _severity(score: float) -> tuple[str, str]:
    if score < 30:
        return "LOW", "RECORD_AND_MONITOR"
    if score < 60:
        return "MEDIUM", "ALERT_FOR_REVIEW"
    if score < 80:
        return "HIGH", "PRIORITIZE_INVESTIGATION"
    return "CRITICAL", "RECOMMEND_CONTAINMENT"


def calculate_risk(data: RiskInput) -> dict[str, Any]:
    """Calculate a 0-100 contextual risk score with an inspectable breakdown."""

    _validate_input(data)
    components = {
        "anomaly_severity": WEIGHTS["anomaly_severity"] * data.anomaly_severity,
        "behavior_deviation": WEIGHTS["behavior_deviation"] * data.behavior_deviation,
        "identity_untrust": WEIGHTS["identity_untrust"] * (1.0 - data.identity_trust),
        "device_untrust": WEIGHTS["device_untrust"] * (1.0 - data.device_trust),
        "asset_criticality": WEIGHTS["asset_criticality"] * data.asset_criticality,
    }
    risk_score = round(100.0 * sum(components.values()), 2)
    risk_level, recommended_action = _severity(risk_score)

    return {
        "identity_id": data.identity_id,
        "device_id": data.device_id,
        "asset_id": data.asset_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "inputs": {
            key: getattr(data, key)
            for key in (
                "identity_trust",
                "device_trust",
                "asset_criticality",
                "anomaly_severity",
                "behavior_deviation",
            )
        },
        "weights": WEIGHTS.copy(),
        "risk_components": {
            name: round(100.0 * value, 2)
            for name, value in components.items()
        },
    }


if __name__ == "__main__":
    example = RiskInput(
        identity_trust=0.85,
        device_trust=0.75,
        asset_criticality=0.80,
        anomaly_severity=0.90,
        behavior_deviation=0.70,
    )
    print(calculate_risk(example))
