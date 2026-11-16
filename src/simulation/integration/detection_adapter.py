"""Detection adapter orchestrating empirical telemetry resolution and detector inference."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.application.detection_service import DetectionService
from src.simulation.data.telemetry_resolver import TelemetryResolver
from src.simulation.models.simulation_event import SimulationEvent


class DetectionAdapter:
    """Invokes existing IT/OT detectors using empirical telemetry without data fabrication."""

    def __init__(self, root: str | Path | None = None, service: DetectionService | None = None) -> None:
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        if service is not None:
            self.service = service
        else:
            self.service = DetectionService(self.root, self.root / "data" / "resinet-events.sqlite3")
        self.resolver = TelemetryResolver(self.root)

    def process_detection(self, event: SimulationEvent, timestamp_dt: datetime) -> dict[str, Any]:
        source_str = event.source.value
        feature_values = self.resolver.resolve(source_str, event.telemetry_record_id)

        # Call existing detection service
        detector = self.service.detector(source_str)
        detect_kwargs = {
            "timestamp": timestamp_dt,
            "asset_id": event.asset_id,
            "device_id": event.device_id,
        }
        raw_result = detector.detect(feature_values, **detect_kwargs)

        # Extract top feature residuals for explainability
        top_features = raw_result.metadata.get("top_contributing_features", [])
        if not top_features and hasattr(detector, "score"):
            # Attempt direct score extraction
            scored = detector.score(feature_values)
            top_features = scored.get("top_contributing_features", [])

        is_anomaly = bool(raw_result.anomaly_decision)
        anomaly_score = float(raw_result.anomaly_score)
        threshold = float(raw_result.threshold) if raw_result.threshold else 0.089945

        # Normalize anomaly severity to [0.0, 1.0]
        if threshold > 0:
            anomaly_severity = min(max(0.0, anomaly_score / threshold), 1.0) if is_anomaly else min(0.3, anomaly_score / threshold)
        else:
            anomaly_severity = 1.0 if is_anomaly else 0.0

        return {
            "detector_name": raw_result.detector,
            "prediction": raw_result.prediction,
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "is_anomaly": is_anomaly,
            "anomaly_severity": anomaly_severity,
            "confidence": float(raw_result.confidence),
            "top_contributing_features": top_features,
            "metadata": raw_result.metadata,
        }

