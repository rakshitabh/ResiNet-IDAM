"""Single orchestration path for IT/OT detection and risk assessment."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.api.database import EventStore
from src.domain.events import DetectionResult, ReviewStatus, SecurityEvent, Source
from src.detectors.registry import build_registry
from src.zero_trust.risk_engine import RiskInput, calculate_risk


class DetectionService:
    def __init__(self, root: Path, database_path: Path) -> None:
        self.root = root
        self.detectors = build_registry(root)
        self.store = EventStore(database_path)

    def detector(self, source: Source) -> Any:
        detector = self.detectors.get(source)
        if detector is None:
            raise ValueError(f"Unsupported detector source: {source}")
        if not detector.available():
            raise RuntimeError(f"{detector.name} artifacts are unavailable")
        return detector

    def availability(self) -> dict[str, bool]:
        return {source: detector.available() for source, detector in self.detectors.items()}

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
    ) -> dict[str, Any]:
        result: DetectionResult = self.detector(source).detect(
            values,
            timestamp=timestamp,
            asset_id=asset_id,
            device_id=device_id,
        )
        anomaly_severity = min(result.anomaly_score / result.threshold, 1.0)
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
            metadata={**result.metadata, "source": result.source},
        )
        return self.store.create_event(event)
