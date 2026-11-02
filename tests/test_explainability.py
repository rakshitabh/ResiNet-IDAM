"""Unit and Integration Tests for Explainable Alerts and SecOps Operations (Phase 6).

Tests:
1. test_autoencoder_explainer_ranking: Verifies element-wise squared reconstruction residuals are ranked correctly.
2. test_fusion_explainer_proportions: Confirms model contributions sum to 100%.
3. test_build_alert_explanation_narrative: Verifies narrative synthesis and structured payload.
4. test_event_store_persists_and_retrieves_explanation: Validates SQLite round-trip for explanation fields.
5. test_secops_review_status_lifecycle: Tests update_review() state transitions ('NEW' -> 'RESOLVED' / 'IN_REVIEW').
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest

from src.api.database import EventStore
from src.domain.events import SecurityEvent
from src.features.explainability import (
    AlertExplanation,
    AutoencoderExplainer,
    FeatureAttribution,
    FusionExplainer,
    build_alert_explanation,
)


def test_autoencoder_explainer_ranking():
    features = ["Sensor_A", "Sensor_B", "Sensor_C", "Sensor_D"]
    x_obs = np.array([1.0, 10.0, 0.5, 3.0])
    x_rec = np.array([1.0, 2.0, 0.5, 3.1])  # Sensor_B has huge error (8^2 = 64)

    explainer = AutoencoderExplainer(features)
    attributions = explainer.explain(x_obs, x_rec, top_k=2)

    assert len(attributions) == 2
    assert attributions[0].feature_name == "Sensor_B"
    assert attributions[0].residual_error == pytest.approx(64.0, abs=1e-3)
    assert attributions[0].observed_value == 10.0
    assert attributions[0].expected_value == 2.0
    assert attributions[1].feature_name == "Sensor_D"


def test_fusion_explainer_proportions():
    weights = {"RF": 0.60, "ResidualAE": 0.20, "IF": 0.20}
    scores = {"RF": 0.80, "ResidualAE": 0.50, "IF": 0.50}

    contribs = FusionExplainer.explain(weights, scores)
    assert sum(contribs.values()) == pytest.approx(100.0, abs=0.1)
    # RF contribution: 0.6*0.8 = 0.48, AE: 0.2*0.5 = 0.10, IF: 0.2*0.5 = 0.10, tot=0.68
    # RF% = 48/68 = 70.59%
    assert contribs["RF"] > contribs["ResidualAE"]
    assert contribs["ResidualAE"] == pytest.approx(contribs["IF"])


def test_build_alert_explanation_narrative():
    explanation = build_alert_explanation(
        domain="OT",
        feature_names=["P1_B4002", "P2_VFD", "P1_LCV"],
        x_observed=np.array([5.0, 1.0, 0.0]),
        x_reconstructed=np.array([1.0, 1.0, 0.0]),
        fusion_weights={"AE": 0.45, "IF": 0.55},
        fusion_scores={"AE": 0.90, "IF": 0.85},
        drift_status="NORMAL",
        threshold_mode="ADAPTIVE",
        top_k=2,
    )
    assert explanation.domain == "OT"
    assert len(explanation.top_features) == 2
    assert explanation.top_features[0].feature_name == "P1_B4002"
    assert "P1_B4002" in explanation.summary_narrative
    assert explanation.drift_status == "NORMAL"


def test_event_store_persists_and_retrieves_explanation(tmp_path: Path):
    db_path = tmp_path / "test_events.sqlite3"
    store = EventStore(db_path)

    event = SecurityEvent(
        event_id="EVT-EXP-001",
        source="OT",
        detector="HAI Autoencoder",
        timestamp=datetime.now(timezone.utc).isoformat(),
        asset_id="P1_Boiler",
        device_id="PLC-01",
        identity_id="operator-admin",
        prediction="ATTACK",
        anomaly_score=1.25,
        threshold=0.82,
        anomaly_decision=True,
        confidence=1.0,
        risk_score=88.5,
        risk_level="CRITICAL",
        review_status="NEW",
        recommended_action="RECOMMEND_CONTAINMENT",
        context={"identity_trust": 0.9, "device_trust": 0.8},
        risk_components={"anomaly_severity": 25.0, "behavior_deviation": 30.0},
        top_contributing_features=[
            {"feature_name": "P1_B4002", "observed_value": 5.0, "expected_value": 1.0, "residual_error": 16.0}
        ],
        fusion_contributions={"AE": 55.0, "IF": 45.0},
        drift_status="NORMAL",
        threshold_mode="ADAPTIVE",
        explanation_summary="Top deviating sensor: P1_B4002.",
        metadata={
            "top_contributing_features": [
                {"feature_name": "P1_B4002", "observed_value": 5.0, "expected_value": 1.0, "residual_error": 16.0}
            ],
            "fusion_contributions": {"AE": 55.0, "IF": 45.0},
            "drift_state": "NORMAL",
            "threshold_mode": "ADAPTIVE",
            "explanation_summary": "Top deviating sensor: P1_B4002.",
        },
    )

    created = store.create_event(event)
    retrieved = store.get("EVT-EXP-001")

    assert retrieved is not None
    assert retrieved["event_id"] == "EVT-EXP-001"
    assert retrieved["risk_level"] == "CRITICAL"
    meta = retrieved["metadata"]
    assert len(meta["top_contributing_features"]) == 1
    assert meta["top_contributing_features"][0]["feature_name"] == "P1_B4002"
    assert meta["fusion_contributions"]["AE"] == 55.0


def test_secops_review_status_lifecycle(tmp_path: Path):
    db_path = tmp_path / "test_lifecycle.sqlite3"
    store = EventStore(db_path)

    event = SecurityEvent(
        event_id="EVT-TRIAGE-001",
        source="IT",
        detector="IDS2025 Random Forest",
        timestamp=datetime.now(timezone.utc).isoformat(),
        asset_id="Web-Server-01",
        device_id="Gateway-01",
        identity_id="svc-account",
        prediction="Botnet ARES",
        anomaly_score=0.95,
        threshold=0.50,
        anomaly_decision=True,
        confidence=0.99,
        risk_score=75.0,
        risk_level="HIGH",
        review_status="NEW",
        recommended_action="PRIORITIZE_INVESTIGATION",
        context={},
        risk_components={},
    )
    store.create_event(event)

    # Operator acknowledges and resolves
    success = store.update_review("EVT-TRIAGE-001", "RESOLVED", "Acknowledged by SecOps")
    assert success
    evt_resolved = store.get("EVT-TRIAGE-001")
    assert evt_resolved["review_status"] == "RESOLVED"

    # Operator escalates
    success_esc = store.update_review("EVT-TRIAGE-001", "IN_REVIEW", "Escalated for containment")
    assert success_esc
    evt_escalated = store.get("EVT-TRIAGE-001")
    assert evt_escalated["review_status"] == "IN_REVIEW"
