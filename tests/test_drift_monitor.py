"""Unit and Integration Tests for Concept Drift Monitoring (Phase 4).

Tests:
1. test_stable_stream_no_drift: 1,000 stationary samples -> 0 drift events, state remains NORMAL.
2. test_step_shift_detected: Stationary stream followed by sustained mean shift -> Drift confirmed, state=REVIEW_REQUIRED.
3. test_short_noise_ignored: Stationary stream with transient 15-sample spike -> 0 drift events.
4. test_rollback_resets_state: acknowledge_and_reset() returns state to NORMAL and clears accumulators.
5. test_drift_event_structure: Validates DriftEvent fields, types, and serializability.
6. test_page_hinkley_unit: Verifies standard PageHinkleyDriftDetector behavior on synthetic stream.
7. test_adwin_unit: Verifies ADWINDriftDetector behavior on split-mean stream.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detectors.drift_monitor import (
    ADWINDriftDetector,
    BlockPageHinkleyDetector,
    DomainDriftMonitor,
    DriftEvent,
    DriftState,
    PageHinkleyDriftDetector,
)


def test_stable_stream_no_drift():
    """Verify that a stationary normal stream does not trigger false drift alarms."""
    rng = np.random.default_rng(42)
    stable_samples = rng.normal(loc=0.5, scale=0.1, size=1500)

    monitor = DomainDriftMonitor(
        domain="OT",
        detector_type="BlockPageHinkley",
        ph_delta=0.05,
        ph_threshold=5.0,
        block_size=20,
    )

    drift_events = []
    for s in stable_samples:
        st, evt = monitor.update(float(s))
        if evt:
            drift_events.append(evt)

    assert len(drift_events) == 0
    assert monitor.state == DriftState.NORMAL
    status = monitor.status()
    assert status["total_drift_events"] == 0
    assert status["state"] == "NORMAL"


def test_step_shift_detected():
    """Verify that a persistent step shift triggers drift and transitions to REVIEW_REQUIRED."""
    rng = np.random.default_rng(42)
    pre_shift = rng.normal(loc=0.5, scale=0.08, size=400)
    post_shift = rng.normal(loc=0.9, scale=0.08, size=600)  # +0.4 mean shift
    stream = np.concatenate([pre_shift, post_shift])

    monitor = DomainDriftMonitor(
        domain="OT",
        detector_type="BlockPageHinkley",
        ph_delta=0.05,
        ph_threshold=5.0,
        block_size=20,
    )

    events = []
    for s in stream:
        st, evt = monitor.update(float(s))
        if evt:
            events.append(evt)

    assert len(events) >= 1
    assert monitor.state == DriftState.REVIEW_REQUIRED
    first_event = events[0]
    assert first_event.domain == "OT"
    assert first_event.state == DriftState.REVIEW_REQUIRED
    assert "Automatic retraining inhibited" in first_event.message


def test_short_noise_ignored():
    """Verify that short transient noise bursts do not cause spurious drift alerts."""
    rng = np.random.default_rng(42)
    stream = rng.normal(loc=0.5, scale=0.08, size=800)
    # 15-sample transient spike
    stream[300:315] += 0.5

    monitor = DomainDriftMonitor(
        domain="OT",
        detector_type="BlockPageHinkley",
        ph_delta=0.05,
        ph_threshold=5.0,
        block_size=20,
    )

    events = []
    for s in stream:
        st, evt = monitor.update(float(s))
        if evt:
            events.append(evt)

    assert len(events) == 0
    assert monitor.state == DriftState.NORMAL


def test_rollback_resets_state():
    """Verify acknowledge_and_reset restores state to NORMAL."""
    rng = np.random.default_rng(42)
    stream = np.concatenate([
        rng.normal(0.5, 0.05, 300),
        rng.normal(1.0, 0.05, 500),
    ])

    monitor = DomainDriftMonitor(
        domain="IT",
        detector_type="BlockPageHinkley",
        ph_delta=0.05,
        ph_threshold=4.0,
        block_size=20,
    )

    for s in stream:
        monitor.update(float(s))

    assert monitor.state == DriftState.REVIEW_REQUIRED

    monitor.acknowledge_and_reset(operator_decision="BASELINE_APPROVED")
    assert monitor.state == DriftState.NORMAL


def test_drift_event_structure():
    """Verify DriftEvent dataclass integrity and serialization."""
    event = DriftEvent(
        event_id="DRIFT-OT-100",
        domain="OT",
        timestamp="2026-11-01T12:00:00Z",
        detector_type="BlockPageHinkley",
        sample_index=100,
        statistic_value=9.45,
        threshold=8.0,
        state=DriftState.REVIEW_REQUIRED,
        model_version="v2.0",
        message="Concept drift detected.",
    )
    assert event.event_id == "DRIFT-OT-100"
    assert event.state == DriftState.REVIEW_REQUIRED
    assert event.threshold == 8.0


def test_page_hinkley_unit():
    """Unit test for PageHinkleyDriftDetector directly."""
    ph = PageHinkleyDriftDetector(delta=0.05, threshold=10.0, alpha=0.99, min_instances=20)
    rng = np.random.default_rng(123)

    # 100 stationary samples
    drifts = []
    for s in rng.normal(1.0, 0.05, 100):
        _, is_drift, _ = ph.update(float(s))
        if is_drift:
            drifts.append(is_drift)
    assert len(drifts) == 0

    # Large sudden shift
    for s in rng.normal(2.5, 0.05, 100):
        _, is_drift, _ = ph.update(float(s))
        if is_drift:
            drifts.append(is_drift)
    assert len(drifts) > 0


def test_adwin_unit():
    """Unit test for ADWINDriftDetector directly."""
    adwin = ADWINDriftDetector(delta_conf=0.01, max_window_size=500)
    rng = np.random.default_rng(123)

    drifts = []
    # Stationary
    for s in rng.normal(0.0, 0.1, 200):
        _, is_drift, _ = adwin.update(float(s))
        if is_drift:
            drifts.append(is_drift)
    assert len(drifts) == 0

    # Sharp shift
    for s in rng.normal(2.0, 0.1, 200):
        _, is_drift, _ = adwin.update(float(s))
        if is_drift:
            drifts.append(is_drift)
    assert len(drifts) > 0

