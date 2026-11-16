"""Simulation run metadata and aggregated metrics model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from src.simulation.models.simulation_result import SimulationEventResult


class RunState(str, Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class SimulationRun:
    run_id: str
    dataset_name: str
    dataset_hash: str
    start_time_iso: str | None = None
    end_time_iso: str | None = None
    state: RunState = RunState.READY
    speed_multiplier: float = 1.0
    elapsed_simulation_seconds: float = 0.0
    active_wall_duration_seconds: float = 0.0
    total_events: int = 0
    processed_events: int = 0
    cancelled_events: int = 0
    failed_events: int = 0
    allow_count: int = 0
    require_verification_count: int = 0
    deny_count: int = 0
    alert_investigation_count: int = 0
    mean_dispatch_jitter_ms: float = 0.0
    mean_processing_latency_ms: float = 0.0
    results: list[SimulationEventResult] = field(default_factory=list)
    error_message: str | None = None

    def update_metrics(self) -> None:
        self.processed_events = len([r for r in self.results if r.status == "COMPLETED"])
        self.failed_events = len([r for r in self.results if r.status == "FAILED"])
        self.cancelled_events = len([r for r in self.results if r.status == "CANCELLED"])
        
        self.allow_count = len([r for r in self.results if r.policy_verdict == "ALLOW"])
        self.require_verification_count = len([r for r in self.results if r.policy_verdict == "REQUIRE VERIFICATION"])
        self.deny_count = len([r for r in self.results if r.policy_verdict == "DENY"])
        self.alert_investigation_count = len([r for r in self.results if r.risk_level in ("HIGH", "CRITICAL")])

        completed = [r for r in self.results if r.status == "COMPLETED"]
        if completed:
            self.mean_dispatch_jitter_ms = sum(r.dispatch_jitter_ms for r in completed) / len(completed)
            self.mean_processing_latency_ms = sum(r.processing_latency_ms for r in completed) / len(completed)

    def to_dict(self) -> dict[str, Any]:
        self.update_metrics()
        return {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "dataset_hash": self.dataset_hash,
            "start_time_iso": self.start_time_iso,
            "end_time_iso": self.end_time_iso,
            "state": self.state.value,
            "speed_multiplier": self.speed_multiplier,
            "elapsed_simulation_seconds": round(self.elapsed_simulation_seconds, 2),
            "active_wall_duration_seconds": round(self.active_wall_duration_seconds, 2),
            "total_events": self.total_events,
            "processed_events": self.processed_events,
            "cancelled_events": self.cancelled_events,
            "failed_events": self.failed_events,
            "verdict_distribution": {
                "ALLOW": self.allow_count,
                "REQUIRE VERIFICATION": self.require_verification_count,
                "DENY": self.deny_count,
            },
            "risk_distribution": {
                "ALERT_INVESTIGATION": self.alert_investigation_count,
            },
            "performance_metrics": {
                "mean_dispatch_jitter_ms": round(self.mean_dispatch_jitter_ms, 3),
                "mean_processing_latency_ms": round(self.mean_processing_latency_ms, 3),
            },
            "results": [r.to_dict() for r in self.results],
            "error_message": self.error_message,
        }

