"""Tabular CSV report generator for simulation execution runs."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from src.simulation.models.simulation_run import SimulationRun


class CsvReportGenerator:
    """Generates a standardized tabular CSV report of all executed, failed, and cancelled events."""

    HEADER = [
        "event_id",
        "source",
        "event_type",
        "user_id",
        "device_id",
        "asset_id",
        "scheduled_offset_sec",
        "scheduled_time",
        "actual_dispatch_time",
        "completion_time",
        "dispatch_jitter_ms",
        "processing_latency_ms",
        "detector_prediction",
        "anomaly_score",
        "detection_threshold",
        "is_anomaly",
        "risk_score",
        "risk_level",
        "policy_verdict",
        "recommended_action",
        "expected_outcome",
        "outcome_matched",
        "status",
        "policy_reason",
    ]

    @classmethod
    def generate(cls, run: SimulationRun) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(cls.HEADER)

        for r in run.results:
            writer.writerow([
                r.event_id,
                r.source,
                r.event_type,
                r.user_id,
                r.device_id,
                r.asset_id,
                r.scheduled_offset,
                r.scheduled_time_iso,
                r.actual_dispatch_time_iso,
                r.completion_time_iso,
                round(r.dispatch_jitter_ms, 3),
                round(r.processing_latency_ms, 3),
                r.detector_prediction,
                round(r.anomaly_score, 6),
                round(r.detection_threshold, 6),
                r.is_anomaly,
                round(r.risk_score, 2),
                r.risk_level,
                r.policy_verdict,
                r.recommended_action,
                r.expected_outcome or "",
                r.outcome_matched if r.outcome_matched is not None else "",
                r.status,
                r.policy_reason,
            ])

        return output.getvalue()

    @classmethod
    def save(cls, run: SimulationRun, target_path: Path) -> Path:
        content = cls.generate(run)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        return target_path

