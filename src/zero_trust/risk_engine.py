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
    "behavior_deviation": 0.30,
    "anomaly_severity": 0.25,
    "asset_criticality": 0.20,
    "identity_untrust": 0.15,
    "device_untrust": 0.10,
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
    if score < 85:
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
    if (
        data.identity_trust <= 0.35
        and data.device_trust <= 0.35
        and data.behavior_deviation >= 0.70
    ):
        risk_level, recommended_action = "CRITICAL", "RECOMMEND_CONTAINMENT"

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


def evaluate_zero_trust_decision(event: dict[str, Any]) -> dict[str, Any]:
    """Evaluate transparent Zero-Trust access rules for decision-support."""
    context = event.get("context", {})
    identity_trust = float(context.get("identity_trust", 1.0))
    device_trust = float(context.get("device_trust", 1.0))
    asset_criticality = float(context.get("asset_criticality", 0.0))
    anomaly_severity = float(context.get("anomaly_severity", 0.0))
    behavior_deviation = float(context.get("behavior_deviation", 0.0))
    risk_score = float(event.get("risk_score", 0.0))

    rule_traces: list[dict[str, str]] = []

    if anomaly_severity >= 0.80 and behavior_deviation >= 0.65 and asset_criticality >= 0.60:
        rule_traces.append(
            {
                "Rule": "R1: Coordinated attack posture",
                "Condition": "anomaly_severity >= 0.80 and behavior_deviation >= 0.65 and asset_criticality >= 0.60",
                "Evaluated": (
                    f"anomaly={anomaly_severity:.2f}, "
                    f"behavior={behavior_deviation:.2f}, "
                    f"asset={asset_criticality:.2f}"
                ),
                "Result": "TRIGGERED",
            }
        )

    if identity_trust < 0.40 and device_trust < 0.40 and risk_score >= 60:
        rule_traces.append(
            {
                "Rule": "R2: Compromised identity and device trust",
                "Condition": "identity_trust < 0.40 and device_trust < 0.40 and risk_score >= 60",
                "Evaluated": (
                    f"identity={identity_trust:.2f}, "
                    f"device={device_trust:.2f}, "
                    f"risk={risk_score:.1f}"
                ),
                "Result": "TRIGGERED",
            }
        )

    if risk_score >= 60 or identity_trust < 0.60 or device_trust < 0.60:
        rule_traces.append(
            {
                "Rule": "R3: Elevated access risk",
                "Condition": "risk_score >= 60 or identity_trust < 0.60 or device_trust < 0.60",
                "Evaluated": (
                    f"risk={risk_score:.1f}, "
                    f"identity={identity_trust:.2f}, "
                    f"device={device_trust:.2f}"
                ),
                "Result": "TRIGGERED",
            }
        )

    contextual_fallback = (
        identity_trust <= 0.35
        and device_trust <= 0.35
        and behavior_deviation >= 0.70
    )
    if contextual_fallback:
        rule_traces.append(
            {
                "Rule": "R4: Contextual trust-behavior fallback",
                "Condition": "identity_trust <= 0.35 and device_trust <= 0.35 and behavior_deviation >= 0.70",
                "Evaluated": (
                    f"identity={identity_trust:.2f}, "
                    f"device={device_trust:.2f}, "
                    f"behavior={behavior_deviation:.2f}"
                ),
                "Result": "TRIGGERED",
            }
        )

    if (
        anomaly_severity >= 0.80
        and behavior_deviation >= 0.65
        and asset_criticality >= 0.60
    ) or (
        identity_trust < 0.40 and device_trust < 0.40 and risk_score >= 60
    ) or contextual_fallback or risk_score >= 85:
        verdict = "DENY"
        action = "REVOKE_AND_ISOLATE"
        reason = (
            "Critical attack posture detected. The event combines abnormal behavior, "
            "high asset criticality, and weakened identity/device trust. Access must be blocked."
        )
    elif risk_score >= 60 or identity_trust < 0.60 or device_trust < 0.60:
        verdict = "REQUIRE VERIFICATION"
        action = "STEP_UP_AUTHENTICATION"
        reason = (
            "Potential security risk detected. Re-authentication, identity validation, "
            "and additional behavioral review are required before access continues."
        )
    else:
        verdict = "ALLOW"
        action = "RECORD_AND_MONITOR"
        reason = (
            "Contextual risk is within acceptable bounds. Identity, device posture, "
            "and behavior indicators are consistent with trusted operations."
        )

    return {
        "verdict": verdict,
        "action": action,
        "recommended_action": action,
        "reason": reason,
        "risk_score": risk_score,
        "rule_traces": rule_traces,
        "is_automatic_action": False,  # Strict boundary: human approval mandatory
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
