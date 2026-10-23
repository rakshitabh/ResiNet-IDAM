import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import main
from src.api.database import EventStore


ROOT = Path(__file__).resolve().parents[1]


def client(tmp_path):
    main.service.store = EventStore(tmp_path / "events.sqlite3")
    return TestClient(main.app)


def valid_payload():
    features = json.loads(
        (ROOT / "models" / "feature_columns.json").read_text(encoding="utf-8")
    )
    return {
        "timestamp": "2022-08-12T16:00:01Z",
        "features": {name: 0.0 for name in features},
        "context": {
            "identity_id": "identity-1",
            "device_id": "device-1",
            "asset_id": "asset-1",
            "identity_trust": 0.8,
            "device_trust": 0.8,
            "asset_criticality": 0.6,
            "behavior_deviation": 0.2,
        },
    }


def test_health_and_model_endpoints(tmp_path):
    response = client(tmp_path).get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    response = client(tmp_path).get("/api/model/info")
    assert response.status_code == 200
    assert response.json()["feature_count"] == 66


def test_score_persist_and_retrieve_event(tmp_path):
    test_client = client(tmp_path)
    response = test_client.post("/api/events/score", json=valid_payload())
    assert response.status_code == 201
    event = response.json()
    assert event["asset_id"] == "asset-1"
    assert 0 <= event["risk_score"] <= 100

    retrieved = test_client.get(f"/api/events/{event['event_id']}")
    assert retrieved.status_code == 200
    assert retrieved.json()["event_id"] == event["event_id"]

    summary = test_client.get("/api/summary")
    assert summary.json()["total_events"] == 1


def test_score_rejects_missing_and_extra_features(tmp_path):
    test_client = client(tmp_path)
    payload = valid_payload()
    payload["features"].pop(next(iter(payload["features"])))
    response = test_client.post("/api/events/score", json=payload)
    assert response.status_code == 422

    payload = valid_payload()
    payload["features"]["unexpected"] = 0.0
    response = test_client.post("/api/events/score", json=payload)
    assert response.status_code == 422


def test_missing_event_returns_not_found(tmp_path):
    response = client(tmp_path).get("/api/events/not-found")
    assert response.status_code == 404
