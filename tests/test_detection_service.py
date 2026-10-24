from pathlib import Path

from src.application.detection_service import DetectionService
from src.infrastructure.config import AppConfig


ROOT = Path(__file__).resolve().parents[1]


def test_registry_reports_available_ot_and_optional_it(tmp_path):
    service = DetectionService(ROOT, tmp_path / "events.sqlite3")
    availability = service.availability()

    assert availability["OT"] is True
    assert availability["IT"] in (True, False)


def test_ot_detection_uses_unified_event_contract(tmp_path):
    service = DetectionService(ROOT, tmp_path / "events.sqlite3")
    schema = service.detector("OT").feature_schema()

    event = service.detect(
        "OT",
        {name: 0.0 for name in schema},
        timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        identity_id="identity-test",
        device_id="device-test",
        asset_id="asset-test",
        identity_trust=0.8,
        device_trust=0.8,
        asset_criticality=0.5,
        behavior_deviation=0.2,
    )

    assert event["source"] == "OT"
    assert event["detector"] == "HAI Autoencoder"
    assert event["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert service.store.get(event["event_id"])["event_id"] == event["event_id"]
