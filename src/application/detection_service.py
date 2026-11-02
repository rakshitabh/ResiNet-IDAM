"""Application orchestration service for intrusion detection workflows."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import inspect
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.api.database import EventStore
from src.detectors.drift_monitor import DomainDriftMonitor, DriftState
from src.detectors.registry import build_registry
from src.domain.events import DetectionResult, ReviewStatus, SecurityEvent, Source
from src.zero_trust.risk_engine import RiskInput, calculate_risk


class DetectionService:
    def __init__(
        self,
        root: str | Path | None = None,
        database_path: str | Path | None = None,
        *,
        registry: Any = None,
        store: EventStore | None = None,
    ) -> None:
        if store is not None:
            self.store = store
        elif database_path is not None:
            self.store = EventStore(Path(database_path))
        else:
            raise ValueError("Either store or database_path must be provided")

        if registry is not None:
            self._registry = registry
        elif root is not None:
            self.root = Path(root)
            self._registry = build_registry(self.root)
        else:
            raise ValueError("Either registry or root must be provided")

        self.drift_monitors: dict[str, DomainDriftMonitor] = {
            "OT": DomainDriftMonitor(domain="OT"),
            "IT": DomainDriftMonitor(domain="IT"),
        }

    def detector(self, source: Source) -> Any:
        if isinstance(self._registry, dict):
            if source in self._registry:
                return self._registry[source]
        if hasattr(self._registry, "get"):
            det = self._registry.get(source)
            if det is not None:
                return det
        raise ValueError(f"Unsupported source: {source}")

    def availability(self) -> dict[str, bool]:
        if hasattr(self._registry, "availability"):
            return self._registry.availability()
        return {
            source: detector.available() if hasattr(detector, "available") else True
            for source, detector in self._registry.items()
        }

    def drift_status(self) -> dict[str, Any]:
        return {domain: monitor.status() for domain, monitor in self.drift_monitors.items()}

    def list_drift_events(self, domain: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.list_drift_events(domain=domain, limit=limit)

    def acknowledge_drift(self, source: str, decision: str = "BASELINE_APPROVED") -> None:
        source = source.upper()
        if source in self.drift_monitors:
            self.drift_monitors[source].acknowledge_and_reset(decision)

    def detect(
        self,
        source: Source,
        values: dict[str, float],
        *,
        timestamp: datetime,
        identity_id: str,
        device_id: str,
        asset_id: str,
        identity_trust: float,
        device_trust: float,
        asset_criticality: float,
        behavior_deviation: float,
        review_status: ReviewStatus = "NEW",
        config_name: str | None = None,
    ) -> dict[str, Any]:
        detector = self.detector(source)

        detect_kwargs: dict[str, Any] = {
            "timestamp": timestamp,
            "asset_id": asset_id,
            "device_id": device_id,
        }
        if config_name is not None and "config_name" in inspect.signature(detector.detect).parameters:
            detect_kwargs["config_name"] = config_name

        result: DetectionResult = detector.detect(values, **detect_kwargs)

        # Monitor concept drift on reconstruction / anomaly score stream
        drift_state = DriftState.NORMAL
        if source in self.drift_monitors:
            drift_state, drift_event = self.drift_monitors[source].update(result.anomaly_score)
            if drift_event:
                self.store.record_drift_event(asdict(drift_event))

        if result.threshold and result.threshold > 0:
            anomaly_severity = min(result.anomaly_score / result.threshold, 1.0)
        else:
            anomaly_severity = 0.0

        risk = calculate_risk(
            RiskInput(
                identity_id=identity_id,
                device_id=device_id,
                asset_id=asset_id,
                identity_trust=identity_trust,
                device_trust=device_trust,
                asset_criticality=asset_criticality,
                anomaly_severity=anomaly_severity,
                behavior_deviation=behavior_deviation,
            )
        )

        top_features = result.metadata.get("top_contributing_features", [])
        fusion_contribs = result.metadata.get("fusion_contributions", {})
        drift_status_str = drift_state.value
        threshold_mode_str = "ADAPTIVE"

        # Generate human-readable narrative explanation
        narrative_parts = []
        if top_features:
            top_names = ", ".join(f"{f['feature_name']}" for f in top_features[:3])
            narrative_parts.append(f"Top deviating telemetry features: {top_names}.")
        if drift_status_str != "NORMAL":
            narrative_parts.append(f"Stream drift state: {drift_status_str}.")
        narrative_parts.append(f"Risk level: {risk['risk_level']} (score: {risk['risk_score']:.1f}). Recommended action: {risk['recommended_action']}.")
        explanation_narrative = " ".join(narrative_parts)

        event = SecurityEvent(
            event_id=str(uuid4()),
            source=result.source,
            detector=result.detector,
            timestamp=result.timestamp.astimezone(timezone.utc).isoformat(),
            asset_id=result.asset_id,
            device_id=result.device_id,
            identity_id=identity_id,
            prediction=result.prediction,
            anomaly_score=result.anomaly_score,
            threshold=result.threshold,
            anomaly_decision=result.anomaly_decision,
            confidence=result.confidence,
            risk_score=risk["risk_score"],
            risk_level=risk["risk_level"],
            review_status=review_status,
            recommended_action=risk["recommended_action"],
            context=risk["inputs"],
            risk_components=risk["risk_components"],
            metadata={
                **result.metadata,
                "source": result.source,
                "drift_state": drift_status_str,
                "top_contributing_features": top_features,
                "fusion_contributions": fusion_contribs,
                "threshold_mode": threshold_mode_str,
                "explanation_summary": explanation_narrative,
            },
            top_contributing_features=top_features,
            fusion_contributions=fusion_contribs,
            drift_status=drift_status_str,
            threshold_mode=threshold_mode_str,
            explanation_summary=explanation_narrative,
        )

        return self.store.create_event(event)
