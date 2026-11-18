"""Simulation run metadata and aggregated metrics model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from src.evaluation.metrics import calculate_classification_metrics, calculate_per_class_recall
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
    experiment_id: str = "LOCO-04"
    domain: str = "IT + OT"
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
    # Research Evaluation Metrics (from image)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    fpr: float = 0.0
    accuracy: float = 0.0
    per_class_recall: dict[str, float] = field(default_factory=dict)
    anomaly_scores_normal: list[float] = field(default_factory=list)
    anomaly_scores_attack: list[float] = field(default_factory=list)
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

            # Update TP/FP/FN/TN
            self.tp = len([r for r in completed if r.classification_result == "TP"])
            self.fp = len([r for r in completed if r.classification_result == "FP"])
            self.fn = len([r for r in completed if r.classification_result == "FN"])
            self.tn = len([r for r in completed if r.classification_result == "TN"])

            metrics = calculate_classification_metrics(self.tp, self.fp, self.fn, self.tn)
            self.precision = metrics["precision"]
            self.recall = metrics["recall"]
            self.f1_score = metrics["f1_score"]
            self.fpr = metrics["fpr"]
            self.accuracy = metrics["accuracy"]

            # Per-class recall
            class_breakdown: dict[str, dict[str, int]] = {}
            for r in completed:
                if r.is_attack_ground_truth:
                    cname = r.ground_truth_label or "Attack"
                    if cname not in class_breakdown:
                        class_breakdown[cname] = {"tp": 0, "fn": 0}
                    if r.classification_result == "TP":
                        class_breakdown[cname]["tp"] += 1
                    elif r.classification_result == "FN":
                        class_breakdown[cname]["fn"] += 1
            self.per_class_recall = calculate_per_class_recall(class_breakdown)

            # Anomaly scores
            self.anomaly_scores_normal = [r.anomaly_score for r in completed if not r.is_attack_ground_truth]
            self.anomaly_scores_attack = [r.anomaly_score for r in completed if r.is_attack_ground_truth]

    def to_dict(self) -> dict[str, Any]:
        self.update_metrics()
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "domain": self.domain,
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
            "classification_metrics": {
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
                "tn": self.tn,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "fpr": self.fpr,
                "accuracy": self.accuracy,
            },
            "per_class_recall": self.per_class_recall,
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

