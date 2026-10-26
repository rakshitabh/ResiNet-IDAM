"""Application orchestration for detector scoring and event persistence."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.api.database import EventStore
from src.detectors.registry import build_registry
from src.domain.events import DetectionResult, ReviewStatus, SecurityEvent, Source
from src.zero_trust.risk_engine import RiskInput, calculate_risk


class DetectionService:
    def __init__(self, root: str | Path, database_path: str | Path) -> None:
        self.root = Path(root)
        self.store = EventStore(Path(database_path))
        self._registry = build_registry(self.root)

    def detector(self, source: Source) -> Any:
        if source not in self._registry:
            raise ValueError(f"Unsupported source: {source}")
        return self._registry[source]

    def availability(self) -> dict[str, bool]:
        return {
            source: detector.available() if hasattr(detector, "available") else True
            for source, detector in self._registry.items()
        }

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