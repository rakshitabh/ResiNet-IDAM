"""Simulation Controller managing state machine, scheduling loop, and adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from src.evaluation.metrics import determine_classification_result
from src.simulation.engine.event_scheduler import EventScheduler
from src.simulation.engine.simulation_clock import SimulationClock
from src.simulation.integration.detection_adapter import DetectionAdapter
from src.simulation.integration.policy_adapter import PolicyAdapter
from src.simulation.integration.risk_adapter import RiskAdapter
from src.simulation.models.simulation_event import EventState, SimulationEvent
from src.simulation.models.simulation_result import SimulationEventResult
from src.simulation.models.simulation_run import RunState, SimulationRun
from src.simulation.persistence.simulation_store import SimulationStore


class SimulationController:
    """Coordinates monotonic simulation time, chronological event execution, and metrics."""

    def __init__(
        self,
        root: str | Path | None = None,
        detection_adapter: DetectionAdapter | None = None,
        store: SimulationStore | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        self.detection_adapter = detection_adapter or DetectionAdapter(self.root)
        self.store = store or SimulationStore(self.root)

        self.clock = SimulationClock()
        self.scheduler: EventScheduler | None = None
        self.current_run: SimulationRun | None = None
        self._raw_events: list[SimulationEvent] = []

    def load_scenario(self, events: list[SimulationEvent], dataset_name: str, dataset_hash: str) -> SimulationRun:
        self._raw_events = [
            SimulationEvent(
                event_id=e.event_id,
                offset_seconds=e.offset_seconds,
                source=e.source,
                event_type=e.event_type,
                user_id=e.user_id,
                device_id=e.device_id,
                asset_id=e.asset_id,
                ip_address=e.ip_address,
                mac_address=e.mac_address,
                telemetry_record_id=e.telemetry_record_id,
                identity_trust=e.identity_trust,
                device_trust=e.device_trust,
                asset_criticality=e.asset_criticality,
                behavior_deviation=e.behavior_deviation,
                expected_outcome=e.expected_outcome,
                ground_truth_label=e.ground_truth_label,
                is_attack_ground_truth=e.is_attack_ground_truth,
                row_number=e.row_number,
                state=EventState.PENDING,
                context=e.context,
            )
            for e in events
        ]
        self.scheduler = EventScheduler(self._raw_events)
        self.clock.reset()

        # Determine domain from events
        sources = {e.source.value for e in self._raw_events}
        domain_str = "IT + OT" if len(sources) > 1 else (next(iter(sources)) if sources else "IT + OT")

        # Extract or infer experiment ID from dataset_name
        exp_id = "LOCO-04"
        if "botnet" in dataset_name.lower():
            exp_id = "LOCO-04 : Botnet ARES (Unseen)"
        elif "infiltrat" in dataset_name.lower():
            exp_id = "LOCO-05 : Infiltration (Unseen)"
        elif "dos" in dataset_name.lower():
            exp_id = "LOCO-01 : DoS/DDoS (Unseen)"
        elif "valve" in dataset_name.lower() or "ot" in dataset_name.lower():
            exp_id = "OT-01 : Actuator Tamper (Unseen)"

        run_id = f"SIM-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
        self.current_run = SimulationRun(
            run_id=run_id,
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
            experiment_id=exp_id,
            domain=domain_str,
            total_events=len(self._raw_events),
            state=RunState.READY,
        )
        return self.current_run

    def start(self) -> None:
        if self.current_run is None or self.scheduler is None:
            raise RuntimeError("No scenario loaded to start")
        if self.current_run.state in (RunState.RUNNING, RunState.COMPLETED, RunState.STOPPED):
            return

        self.clock.start()
        self.current_run.start_time_iso = self.clock.wall_start_iso
        self.current_run.state = RunState.RUNNING

    def pause(self) -> None:
        if self.current_run is not None and self.current_run.state == RunState.RUNNING:
            self.clock.pause()
            self.current_run.state = RunState.PAUSED

    def resume(self) -> None:
        if self.current_run is not None and self.current_run.state == RunState.PAUSED:
            self.clock.resume()
            self.current_run.state = RunState.RUNNING

    def stop(self) -> None:
        if self.current_run is not None and self.current_run.state in (RunState.RUNNING, RunState.PAUSED):
            self.current_run.state = RunState.STOPPING
            self.clock.stop()

            # Cancel remaining pending events and record them in results for reporting (Rule 8)
            if self.scheduler:
                cancelled = self.scheduler.cancel_pending()
                for c_ev in cancelled:
                    cancel_res = SimulationEventResult(
                        event_id=c_ev.event_id,
                        source=c_ev.source.value,
                        event_type=c_ev.event_type,
                        user_id=c_ev.user_id,
                        device_id=c_ev.device_id,
                        asset_id=c_ev.asset_id,
                        scheduled_offset=c_ev.offset_seconds,
                        scheduled_time_iso=self.clock.format_time(c_ev.offset_seconds),
                        actual_dispatch_time_iso=datetime.now(timezone.utc).isoformat(),
                        completion_time_iso=datetime.now(timezone.utc).isoformat(),
                        dispatch_jitter_ms=0.0,
                        processing_latency_ms=0.0,
                        detector_prediction="NOT_EXECUTED",
                        anomaly_score=0.0,
                        detection_threshold=0.0,
                        is_anomaly=False,
                        risk_score=0.0,
                        risk_level="CANCELLED",
                        policy_verdict="CANCELLED",
                        recommended_action="NONE",
                        policy_reason="Simulation stopped before scheduled execution",
                        expected_outcome=c_ev.expected_outcome,
                        status="CANCELLED",
                    )
                    self.current_run.results.append(cancel_res)

            self.current_run.end_time_iso = datetime.now(timezone.utc).isoformat()
            self.current_run.elapsed_simulation_seconds = self.clock.elapsed_sim_seconds()
            self.current_run.active_wall_duration_seconds = self.clock.active_wall_seconds()
            self.current_run.state = RunState.STOPPED
            self.current_run.update_metrics()
            self.store.save_run(self.current_run)

    def restart(self) -> SimulationRun:
        if self.current_run is not None:
            # Archive previous run if not already archived
            self.store.save_run(self.current_run)
            dataset_name = self.current_run.dataset_name
            dataset_hash = self.current_run.dataset_hash
        else:
            dataset_name = "unknown"
            dataset_hash = "unknown"

        # Re-arm fresh events with clean state
        return self.load_scenario(self._raw_events, dataset_name, dataset_hash)

    def set_speed(self, multiplier: float) -> None:
        self.clock.set_speed(multiplier)
        if self.current_run:
            self.current_run.speed_multiplier = float(multiplier)

    def _execute_event(self, event: SimulationEvent, sim_elapsed: float) -> SimulationEventResult:
        event.state = EventState.RUNNING
        t_dispatch_mono = time.monotonic()
        t_dispatch_dt = datetime.now(timezone.utc)

        # Dispatch Jitter: difference between actual simulation elapsed time and scheduled offset
        jitter_ms = max(0.0, (sim_elapsed - event.offset_seconds) * 1000.0)

        # 1. Detection via DetectionAdapter
        det_result = self.detection_adapter.process_detection(event, t_dispatch_dt)

        # 2. 5-Factor Zero-Trust Risk Assessment
        risk_result = RiskAdapter.evaluate_risk(event, det_result["anomaly_severity"])

        # 3. Authoritative Policy Decision
        policy_result = PolicyAdapter.evaluate_policy(risk_result)

        t_complete_mono = time.monotonic()
        t_complete_dt = datetime.now(timezone.utc)
        proc_latency_ms = max(0.0, (t_complete_mono - t_dispatch_mono) * 1000.0)

        # Scientific evaluation against isolated ground truth (Rule 4)
        classification_res = determine_classification_result(
            is_predicted_anomaly=det_result["is_anomaly"],
            is_actual_attack=event.is_attack_ground_truth,
        )

        # Compare with expected outcome if provided
        outcome_matched = None
        if event.expected_outcome:
            clean_expected = event.expected_outcome.strip().upper().replace(" ", "_")
            clean_verdict = policy_result["verdict"].strip().upper().replace(" ", "_")
            outcome_matched = (clean_expected == clean_verdict)

        event.state = EventState.COMPLETED

        result = SimulationEventResult(
            event_id=event.event_id,
            source=event.source.value,
            event_type=event.event_type,
            user_id=event.user_id,
            device_id=event.device_id,
            asset_id=event.asset_id,
            scheduled_offset=event.offset_seconds,
            scheduled_time_iso=self.clock.format_time(event.offset_seconds),
            actual_dispatch_time_iso=t_dispatch_dt.isoformat(),
            completion_time_iso=t_complete_dt.isoformat(),
            dispatch_jitter_ms=jitter_ms,
            processing_latency_ms=proc_latency_ms,
            detector_prediction=det_result["prediction"],
            anomaly_score=det_result["anomaly_score"],
            detection_threshold=det_result["threshold"],
            is_anomaly=det_result["is_anomaly"],
            risk_score=risk_result["risk_score"],
            risk_level=risk_result["risk_level"],
            policy_verdict=policy_result["verdict"],
            recommended_action=policy_result["recommended_action"],
            policy_reason=policy_result["reason"],
            top_contributing_features=det_result["top_contributing_features"],
            rule_traces=policy_result["rule_traces"],
            risk_components=risk_result["risk_components"],
            expected_outcome=event.expected_outcome,
            outcome_matched=outcome_matched,
            ground_truth_label=event.ground_truth_label,
            is_attack_ground_truth=event.is_attack_ground_truth,
            classification_result=classification_res,
            status="COMPLETED",
        )
        return result

    def tick(self) -> list[SimulationEventResult]:
        """Poll the clock and process any due events."""
        if self.current_run is None or self.current_run.state != RunState.RUNNING or self.scheduler is None:
            return []

        sim_elapsed = self.clock.elapsed_sim_seconds()
        self.current_run.elapsed_simulation_seconds = sim_elapsed
        self.current_run.active_wall_duration_seconds = self.clock.active_wall_seconds()

        due_events = self.scheduler.get_due_events(sim_elapsed)
        new_results: list[SimulationEventResult] = []

        for event in due_events:
            try:
                res = self._execute_event(event, sim_elapsed)
                self.current_run.results.append(res)
                new_results.append(res)
            except Exception as ex:
                event.state = EventState.FAILED
                fail_res = SimulationEventResult(
                    event_id=event.event_id,
                    source=event.source.value,
                    event_type=event.event_type,
                    user_id=event.user_id,
                    device_id=event.device_id,
                    asset_id=event.asset_id,
                    scheduled_offset=event.offset_seconds,
                    scheduled_time_iso=self.clock.format_time(event.offset_seconds),
                    actual_dispatch_time_iso=datetime.now(timezone.utc).isoformat(),
                    completion_time_iso=datetime.now(timezone.utc).isoformat(),
                    dispatch_jitter_ms=0.0,
                    processing_latency_ms=0.0,
                    detector_prediction="ERROR",
                    anomaly_score=0.0,
                    detection_threshold=0.0,
                    is_anomaly=False,
                    risk_score=0.0,
                    risk_level="UNKNOWN",
                    policy_verdict="ERROR",
                    recommended_action="NONE",
                    policy_reason=f"Processing exception: {ex}",
                    expected_outcome=event.expected_outcome,
                    error_message=str(ex),
                    status="FAILED",
                )
                self.current_run.results.append(fail_res)
                new_results.append(fail_res)

        self.current_run.update_metrics()

        # Check if all events completed
        if self.scheduler.is_finished():
            self.clock.stop()
            self.current_run.state = RunState.COMPLETED
            self.current_run.end_time_iso = datetime.now(timezone.utc).isoformat()
            self.store.save_run(self.current_run)

        return new_results

    def run_to_completion(self) -> SimulationRun:
        """Execute all scheduled events sequentially without sleep (ideal for testing and batch evaluations)."""
        self.start()
        if self.scheduler is None:
            raise RuntimeError("Scheduler not initialized")

        for event in self.scheduler.events:
            res = self._execute_event(event, event.offset_seconds)
            self.current_run.results.append(res)

        self.clock.stop()
        if self.current_run:
            self.current_run.state = RunState.COMPLETED
            self.current_run.end_time_iso = datetime.now(timezone.utc).isoformat()
            self.current_run.elapsed_simulation_seconds = max((e.offset_seconds for e in self.scheduler.events), default=0.0)
            self.current_run.active_wall_duration_seconds = self.clock.active_wall_seconds()
            self.current_run.update_metrics()
            self.store.save_run(self.current_run)
            return self.current_run
        raise RuntimeError("Current run disappeared")

