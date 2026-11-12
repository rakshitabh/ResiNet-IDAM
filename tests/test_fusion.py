"""Unit and Integration Tests for Calibrated Multi-Model Fusion (Phase 5).

Tests:
1. test_sigmoid_calibration_properties: Verifies centered sigmoid maps threshold -> 0.50, is monotonic, bounded in [0, 1].
2. test_it_fusion_detector_static: Validates weighted combination, weight normalization, and anomaly decision.
3. test_it_fusion_zero_day_safeguard: Confirms dynamic boost of unsupervised weights during zero-day divergence.
4. test_ot_fusion_detector_static: Confirms empirical OT weights (AE=0.45, IF=0.55) and output structure.
5. test_feedback_weight_update: Verifies feedback updates reweight models proportionally to reliability.
6. test_zero_trust_risk_alignment: Confirms FusedDetectionResult feeds into ZeroTrustRiskEngine correctly.
7. test_contribution_decomposition: Verifies sum of model contributions matches fused score.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detectors.fusion import (
    FusionMode,
    ITFusionDetector,
    OTFusionDetector,
    evaluate_fused_risk,
    sigmoid_calibrate,
    to_zero_trust_risk_input,
)


def test_sigmoid_calibration_properties():
    # Value at threshold must map to exactly 0.50
    thresh = 0.75
    s_mid = sigmoid_calibrate(thresh, threshold=thresh, gamma=5.0)
    assert abs(s_mid - 0.50) < 1e-5

    # Strictly monotonic
    s_low = sigmoid_calibrate(thresh - 0.20, threshold=thresh, gamma=5.0)
    s_high = sigmoid_calibrate(thresh + 0.20, threshold=thresh, gamma=5.0)
    assert s_low < 0.50 < s_high
    assert 0.0 <= s_low <= 1.0
    assert 0.0 <= s_high <= 1.0


def test_it_fusion_detector_static():
    detector = ITFusionDetector(weights=(0.60, 0.20, 0.20), decision_threshold=0.50)
    assert sum(detector.weights.values()) == pytest.approx(1.0)
    assert detector.weights["RF"] == pytest.approx(0.60)
    assert detector.weights["ResidualAE"] == pytest.approx(0.20)
    assert detector.weights["IF"] == pytest.approx(0.20)

    # Benign sample
    res_benign = detector.fuse(rf_prob_attack=0.01, ae_mse=0.10, if_raw_score=0.20)
    assert res_benign.domain == "IT"
    assert res_benign.fused_score < 0.50
    assert not res_benign.is_anomaly
    assert not res_benign.zero_day_suspected

    # Severe attack sample
    res_attack = detector.fuse(rf_prob_attack=0.99, ae_mse=1.50, if_raw_score=0.80)
    assert res_attack.fused_score > 0.80
    assert res_attack.is_anomaly


def test_it_fusion_zero_day_safeguard():
    # In adaptive mode, if RF says benign (0.05) but AE and IF observe extreme deviations (0.95),
    # zero-day safeguard must activate
    detector = ITFusionDetector(
        weights=(0.60, 0.20, 0.20),
        decision_threshold=0.50,
        mode=FusionMode.ADAPTIVE,
    )
    res = detector.fuse(
        rf_prob_attack=0.05,
        ae_mse=2.50,  # extreme reconstruction error
        if_raw_score=0.85,  # extreme density outlier
        enable_zero_day_boost=True,
    )
    assert res.zero_day_suspected
    assert res.weights["ResidualAE"] > detector.base_weights["ResidualAE"]
    assert res.weights["IF"] > detector.base_weights["IF"]
    assert res.is_anomaly  # elevated to anomaly despite RF missing it


def test_ot_fusion_detector_static():
    ot_det = OTFusionDetector(weights=(0.45, 0.55), decision_threshold=0.50)
    assert sum(ot_det.weights.values()) == pytest.approx(1.0)

    res = ot_det.fuse(ae_mse=0.821954, if_raw_score=0.5284)
    # Both are at their respective thresholds -> calibrated to ~0.50 each
    assert abs(res.fused_score - 0.50) < 0.05
    assert "AE" in res.contributions
    assert "IF" in res.contributions


def test_feedback_weight_update():
    detector = ITFusionDetector(weights=(0.33, 0.33, 0.34))
    # Synthetic verified batch where AE is perfect and RF/IF make errors
    verified = [
        {"y_true": 1, "rf_pred": 0, "ae_pred": 1, "if_pred": 0},
        {"y_true": 1, "rf_pred": 0, "ae_pred": 1, "if_pred": 0},
        {"y_true": 0, "rf_pred": 1, "ae_pred": 0, "if_pred": 1},
    ]
    detector.update_weights_from_feedback(verified, alpha=0.50)
    assert sum(detector.weights.values()) == pytest.approx(1.0)
    # AE weight should have increased due to 0 error
    assert detector.weights["ResidualAE"] > detector.weights["RF"]
    assert detector.weights["ResidualAE"] > detector.weights["IF"]


def test_zero_trust_risk_alignment():
    ot_det = OTFusionDetector()
    res = ot_det.fuse(ae_mse=1.80, if_raw_score=0.75)

    risk_output = evaluate_fused_risk(
        res,
        identity_trust=0.90,
        device_trust=0.80,
        asset_criticality=0.95,
    )
    assert "risk_score" in risk_output
    assert "risk_level" in risk_output
    assert "fusion_detail" in risk_output
    assert 0.0 <= risk_output["risk_score"] <= 100.0
    assert risk_output["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert risk_output["inputs"]["anomaly_severity"] == pytest.approx(res.fused_score, abs=1e-3)


def test_contribution_decomposition():
    detector = ITFusionDetector(weights=(0.50, 0.30, 0.20))
    res = detector.fuse(rf_prob_attack=0.80, ae_mse=0.70, if_raw_score=0.50)
    # Sum of contributions must equal fused_score
    total_contrib = sum(res.contributions.values())
    assert abs(total_contrib - res.fused_score) < 1e-5

