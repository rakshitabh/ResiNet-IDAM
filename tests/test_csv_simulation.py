"""Comprehensive functional and integration test suite for CSV-driven Zero-Trust Simulation (TC-01 to TC-12)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import time

from src.simulation.data.csv_parser import SimulationCsvParser
from src.simulation.data.dataset_validator import (
    MAX_CSV_BYTES,
    MAX_EVENT_ROWS,
    ValidationError,
)
from src.simulation.engine.controller import SimulationController
from src.simulation.models.simulation_event import EventSource, EventState, SimulationEvent
from src.simulation.models.simulation_run import RunState
from src.simulation.reporting.csv_report import CsvReportGenerator
from src.simulation.reporting.json_report import JsonReportGenerator


ROOT = Path(__file__).resolve().parents[1]


# TC-01: Missing columns or invalid CSV
def test_tc01_missing_columns_or_invalid_csv():
    # Missing required 'asset_id'
    bad_csv = "event_id,offset_seconds,source,event_type,user_id,device_id\nEVT1,2.0,OT,LOGIN,U1,D1\n"
    with pytest.raises(ValidationError) as exc:
        SimulationCsvParser.parse(bad_csv)
    assert "Missing mandatory columns" in str(exc.value)

    # Invalid source
    bad_source = "event_id,offset_seconds,source,event_type,user_id,device_id,asset_id\nEVT1,2.0,CLOUD,LOGIN,U1,D1,A1\n"
    with pytest.raises(ValidationError) as exc2:
        SimulationCsvParser.parse(bad_source)
    assert "source must be 'IT' or 'OT'" in str(exc2.value)


# TC-02: Upload exceeds limits
def test_tc02_upload_exceeds_limits():
    oversized_bytes = b"x" * (MAX_CSV_BYTES + 1024)
    with pytest.raises(ValidationError) as exc:
        SimulationCsvParser.parse(oversized_bytes)
    assert "exceeds maximum permitted limit" in str(exc.value)


# TC-03: Duplicate event IDs
def test_tc03_duplicate_event_ids():
    dup_csv = (
        "event_id,offset_seconds,source,event_type,user_id,device_id,asset_id\n"
        "EVT001,1.0,OT,LOGIN,U1,D1,A1\n"
        "EVT001,3.0,OT,ACCESS,U1,D1,A1\n"
    )
    with pytest.raises(ValidationError) as exc:
        SimulationCsvParser.parse(dup_csv)
    assert "Duplicate event_id 'EVT001'" in str(exc.value)


# TC-04: Equal event offsets preserve original order
def test_tc04_equal_event_offsets():
    equal_offset_csv = (
        "event_id,offset_seconds,source,event_type,user_id,device_id,asset_id\n"
        "EVT_B,4.0,OT,READ,U1,D1,A1\n"
        "EVT_A,4.0,OT,WRITE,U1,D1,A1\n"
        "EVT_C,4.0,OT,ALERT,U1,D1,A1\n"
    )
    events, _ = SimulationCsvParser.parse(equal_offset_csv)
    assert [e.event_id for e in events] == ["EVT_B", "EVT_A", "EVT_C"]


# TC-05: Pause and resume preserves timing
def test_tc05_pause_and_resume():
    controller = SimulationController(root=ROOT)
    scenario_path = ROOT / "data" / "simulation" / "scenarios" / "zero_trust_simulation_scenarios.csv"
    events, d_hash = SimulationCsvParser.parse(scenario_path)

    controller.load_scenario(events, "test_pause", d_hash)
    controller.start()
    assert controller.current_run.state == RunState.RUNNING

    controller.pause()
    assert controller.current_run.state == RunState.PAUSED
    time.sleep(0.1)
    paused_elapsed = controller.clock.elapsed_sim_seconds()
    time.sleep(0.1)
    # Elapsed time must not advance while paused
    assert controller.clock.elapsed_sim_seconds() == paused_elapsed

    controller.resume()
    assert controller.current_run.state == RunState.RUNNING
    controller.stop()


# TC-06: Stop with pending events cancels pending
def test_tc06_stop_with_pending_events():
    controller = SimulationController(root=ROOT)
    scenario_path = ROOT / "data" / "simulation" / "scenarios" / "zero_trust_simulation_scenarios.csv"
    events, d_hash = SimulationCsvParser.parse(scenario_path)

    controller.load_scenario(events, "test_stop", d_hash)
    controller.start()
    controller.stop()

    assert controller.current_run.state == RunState.STOPPED
    assert controller.current_run.cancelled_events > 0
    assert all(e.state == EventState.CANCELLED for e in controller.scheduler.events)


# TC-07: Restart generates fresh run ID and preserves history
def test_tc07_restart():
    controller = SimulationController(root=ROOT)
    scenario_path = ROOT / "data" / "simulation" / "scenarios" / "normal_it.csv"
    events, d_hash = SimulationCsvParser.parse(scenario_path)

    controller.load_scenario(events, "test_restart", d_hash)
    first_run_id = controller.current_run.run_id
    controller.start()
    controller.stop()

    fresh_run = controller.restart()
    second_run_id = fresh_run.run_id
    assert first_run_id != second_run_id
    assert fresh_run.state == RunState.READY
    assert fresh_run.total_events == len(events)
    # Old run was saved in runs
    saved_old = controller.store.get_run(first_run_id)
    assert saved_old is not None


# TC-08: Expected label independence (Rule 4)
def test_tc08_expected_label_independence():
    controller = SimulationController(root=ROOT)
    # Malicious parameters with expected_outcome='ALLOW'
    event1 = SimulationEvent(
        event_id="EVT_MAL_1",
        offset_seconds=1.0,
        source=EventSource.OT,
        event_type="SETPOINT_OVERRIDE",
        user_id="USR_MAL",
        device_id="DEV_ROGUE",
        asset_id="TURBINE_P4",
        identity_trust=0.10,
        device_trust=0.10,
        asset_criticality=0.95,
        behavior_deviation=0.95,
        expected_outcome="ALLOW",  # Intentional ground truth contradiction
    )

    controller.load_scenario([event1], "test_label_indep", "dummy_hash")
    run = controller.run_to_completion()
    result = run.results[0]

    # Detector and risk engine must evaluate empirical risk, ignoring expected_outcome
    assert result.risk_score >= 60.0
    assert result.policy_verdict == "DENY"
    assert result.outcome_matched is False


# TC-09: Risk scale is strictly within 0-100
def test_tc09_risk_scale():
    controller = SimulationController(root=ROOT)
    scenario_path = ROOT / "data" / "simulation" / "scenarios" / "mixed_it_ot.csv"
    events, d_hash = SimulationCsvParser.parse(scenario_path)

    controller.load_scenario(events, "test_risk_scale", d_hash)
    run = controller.run_to_completion()

    for r in run.results:
        assert 0.0 <= r.risk_score <= 100.0
        assert r.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


# TC-10: Correct detector routing for IT and OT
def test_tc10_detector_routing():
    controller = SimulationController(root=ROOT)
    scenario_path = ROOT / "data" / "simulation" / "scenarios" / "mixed_it_ot.csv"
    events, d_hash = SimulationCsvParser.parse(scenario_path)

    controller.load_scenario(events, "test_routing", d_hash)
    run = controller.run_to_completion()

    it_results = [r for r in run.results if r.source == "IT"]
    ot_results = [r for r in run.results if r.source == "OT"]

    assert len(it_results) > 0
    assert len(ot_results) > 0
    for r in ot_results:
        # OT uses HAI Autoencoder
        assert "Autoencoder" in r.detector_prediction or r.detector_prediction in ("NORMAL", "ATTACK")
    for r in it_results:
        # IT uses IDS2025
        assert r.detector_prediction in ("Normal", "Attack", "ATTACK", "NORMAL")


# TC-11: Safety boundary enforcement (Rule 10)
def test_tc11_safety_boundary():
    controller = SimulationController(root=ROOT)
    scenario_path = ROOT / "data" / "simulation" / "scenarios" / "ot_anomaly.csv"
    events, d_hash = SimulationCsvParser.parse(scenario_path)

    controller.load_scenario(events, "test_safety", d_hash)
    run = controller.run_to_completion()

    for r in run.results:
        # All actions must be advisory
        assert "Simulated Advisory" in r.recommended_action or r.policy_verdict == "ALLOW"


# TC-12: Report reconciliation
def test_tc12_report_reconciliation():
    controller = SimulationController(root=ROOT)
    scenario_path = ROOT / "data" / "simulation" / "scenarios" / "zero_trust_simulation_scenarios.csv"
    events, d_hash = SimulationCsvParser.parse(scenario_path)

    controller.load_scenario(events, "test_recon", d_hash)
    run = controller.run_to_completion()

    csv_text = CsvReportGenerator.generate(run)
    json_text = JsonReportGenerator.generate(run)

    json_obj = json.loads(json_text)
    assert json_obj["run_id"] == run.run_id
    assert json_obj["total_events"] == run.total_events
    assert len(json_obj["results"]) == len(run.results)
    assert "output_report_sha256" in json_obj["audit_metadata"]

    lines = csv_text.strip().split("\n")
    assert len(lines) == len(run.results) + 1  # header + results

