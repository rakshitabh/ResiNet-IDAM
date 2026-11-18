"""Risk assessment adapter connecting simulation context to the 5-factor Zero-Trust engine."""

from __future__ import annotations

from typing import Any
from src.simulation.models.simulation_event import SimulationEvent
from src.zero_trust.risk_engine import RiskInput, calculate_risk


class RiskAdapter:
    """Invokes the existing 5-factor Zero-Trust Risk Engine without formula duplication."""

    @classmethod
    def evaluate_risk(cls, event: SimulationEvent, anomaly_severity: float) -> dict[str, Any]:
        risk_input = RiskInput(
            identity_id=event.user_id,
            device_id=event.device_id,
            asset_id=event.asset_id,
            identity_trust=event.identity_trust,
            device_trust=event.device_trust,
            asset_criticality=event.asset_criticality,
            anomaly_severity=anomaly_severity,
            behavior_deviation=event.behavior_deviation,
        )
        return calculate_risk(risk_input)

