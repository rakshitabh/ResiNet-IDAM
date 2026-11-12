"""Unit tests for Phase 3 Adaptive Threshold Engine with Poisoning Safeguards."""

import pytest
import numpy as np
from src.detectors.adaptive_threshold import (
    AdaptiveThresholdEngine,
    ThresholdStatus,
)


def test_initialization():
    engine = AdaptiveThresholdEngine(base_threshold=1.0, k=3.0, window_size=50)
    assert engine.base_threshold == 1.0
    assert engine.current_threshold == 1.0
    assert engine.buffer_count == 0


def test_formula_recalculation():
    # If buffer has values [0.1, 0.2, 0.1, 0.2]
    # mu = 0.15, sigma = 0.05
    # T = mu + 3 * sigma = 0.15 + 0.15 = 0.30
    engine = AdaptiveThresholdEngine(
        base_threshold=0.5,
        k=3.0,
        window_size=10,
        min_buffer_size=4,
        max_relative_change=1.0,  # allow full change
    )

    for val in [0.1, 0.2, 0.1, 0.2]:
        engine.evaluate_and_update(val)

    # Threshold should have adapted to ~0.30
    assert abs(engine.current_threshold - 0.30) < 0.01
    assert engine.buffer_count == 4


def test_anti_poisoning_buffer_exclusion():
    """Anomalies above threshold and high risk scores must be excluded from buffer."""
    engine = AdaptiveThresholdEngine(
        base_threshold=1.0,
        k=3.0,
        window_size=20,
        min_buffer_size=5,
    )

    # Feed 4 benign values
    for val in [0.2, 0.2, 0.2, 0.2]:
        is_anom, _ = engine.evaluate_and_update(val)
        assert not is_anom
    assert engine.buffer_count == 4

    # Feed an anomaly (> 1.0)
    is_anom, _ = engine.evaluate_and_update(10.0)
    assert is_anom
    # Must NOT have been added to buffer!
    assert engine.buffer_count == 4

    # Feed a high-risk sample (> 40.0 risk score)
    is_anom, _ = engine.evaluate_and_update(0.3, risk_score=85.0)
    # Must NOT have been added to buffer
    assert engine.buffer_count == 4


def test_max_relative_change_clamping():
    """Threshold changes per window must be clamped to max_relative_change limit."""
    engine = AdaptiveThresholdEngine(
        base_threshold=2.0,
        k=3.0,
        window_size=10,
        min_buffer_size=3,
        max_relative_change=0.05,  # Max 5% change per step (max change = 0.10)
    )

    # Feed values below base_threshold (2.0) with high variance so raw_t is ~2.18 (delta = 0.18 > 0.10)
    for val in [0.8, 1.6, 1.2]:
        engine.evaluate_and_update(val)

    # Base was 2.0, maximum single update jump is 2.0 * 0.05 = 0.10 -> applied_t = 2.10
    assert engine.current_threshold <= 2.1001
    summary = engine.status_summary()
    assert summary["latest_status"] == ThresholdStatus.CLAMPED.value


def test_rollback_mechanism():
    engine = AdaptiveThresholdEngine(
        base_threshold=1.0,
        k=3.0,
        window_size=10,
        min_buffer_size=2,
    )
    engine.evaluate_and_update(0.5)
    engine.evaluate_and_update(0.5)
    assert engine.buffer_count > 0

    engine.reset_to_base(reason="Emergency Operator Rollback")
    assert engine.current_threshold == 1.0
    assert engine.buffer_count == 0
    assert engine.status_summary()["latest_status"] == ThresholdStatus.ROLLED_BACK.value
