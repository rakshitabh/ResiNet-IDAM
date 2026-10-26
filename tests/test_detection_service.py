from pathlib import Path

from src.application.detection_service import DetectionService
from src.infrastructure.config import AppConfig


ROOT = Path(__file__).resolve().parents[1]


def test_registry_reports_available_ot_and_optional_it(tmp_path):
    service = DetectionService(ROOT, tmp_path / "events.sqlite3")
    availability = service.availability()

    assert availability["OT"] is True
    assert availability["IT"] is True


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


def test_alert_lifecycle_status_transitions(tmp_path):
    import pytest
    service = DetectionService(ROOT, tmp_path / "events.sqlite3")
    schema = service.detector("OT").feature_schema()

    event = service.detect(
        "OT",
        {name: 0.0 for name in schema},
        timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        identity_id="identity-lifecycle",
        device_id="device-lifecycle",
        asset_id="asset-lifecycle",
        identity_trust=0.8,
        device_trust=0.8,
        asset_criticality=0.5,
        behavior_deviation=0.2,
    )
    event_id = event["event_id"]
    assert service.store.get(event_id)["review_status"] == "NEW"

    # Transition to IN_REVIEW
    assert service.store.update_review(event_id, "IN_REVIEW")
    assert service.store.get(event_id)["review_status"] == "IN_REVIEW"

    # Transition to RESOLVED
    assert service.store.update_review(event_id, "RESOLVED")
    assert service.store.get(event_id)["review_status"] == "RESOLVED"


def test_missing_features_and_incorrect_order_rejected(tmp_path):
    import pytest
    service = DetectionService(ROOT, tmp_path / "events.sqlite3")
    schema = service.detector("OT").feature_schema()

    # Missing features
    incomplete_features = {schema[0]: 0.0}
    with pytest.raises(ValueError):
        service.detect(
            "OT",
            incomplete_features,
            timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            identity_id="id-1",
            device_id="dev-1",
            asset_id="ast-1",
            identity_trust=0.8,
            device_trust=0.8,
            asset_criticality=0.5,
            behavior_deviation=0.2,
        )

    # Incorrect ordering (reverse keys)
    reversed_features = {k: 0.0 for k in reversed(schema)}
    with pytest.raises(ValueError):
        service.detect(
            "OT",
            reversed_features,
            timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            identity_id="id-1",
            device_id="dev-1",
            asset_id="ast-1",
            identity_trust=0.8,
            device_trust=0.8,
            asset_criticality=0.5,
            behavior_deviation=0.2,
        )


def test_unconfigured_detector_fails_safely(tmp_path):
    import pytest
    service = DetectionService(ROOT, tmp_path / "events.sqlite3")
    with pytest.raises((ValueError, KeyError)):
        service.detector("NON_EXISTENT")


def test_it_detection_uses_unified_event_contract(tmp_path):
    service = DetectionService(ROOT, tmp_path / "events.sqlite3")
    schema = service.detector("IT").feature_schema("all_features")

    event = service.detect(
        "IT",
        {name: 0.0 for name in schema},
        timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        identity_id="identity-it-test",
        device_id="device-it-test",
        asset_id="asset-it-test",
        identity_trust=0.85,
        device_trust=0.9,
        asset_criticality=0.7,
        behavior_deviation=0.1,
        config_name="all_features",
    )

    assert event["source"] == "IT"
    assert event["detector"] == "IDS2025 Random Forest"
    assert event["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert service.store.get(event["event_id"])["event_id"] == event["event_id"]
