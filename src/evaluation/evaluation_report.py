"""Evaluation report generation engine for JSON and CSV exports (Sprint 4 & Sprint 6)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from src.evaluation.provenance import ProvenanceTracker


class EvaluationReportGenerator:
    """Generates structured JSON, CSV, and summary reports with complete cryptographic provenance."""

    @classmethod
    def generate_json_report(
        cls,
        experiment_id: str,
        provenance: dict[str, Any],
        evaluation_results: dict[str, Any],
    ) -> str:
        """Generate structured JSON experiment report."""
        report_data = {
            "experiment_id": experiment_id,
            "provenance": provenance,
            "metrics": evaluation_results.get("metrics", {}),
            "per_class_recall": evaluation_results.get("per_class_recall", {}),
            "unseen_class_recall": evaluation_results.get("unseen_class_recall", {}),
            "latency": evaluation_results.get("latency", {}),
            "sample_count": len(evaluation_results.get("sample_results", [])),
            "sample_results": [
                s.to_dict() if hasattr(s, "to_dict") else s
                for s in evaluation_results.get("sample_results", [])
            ],
        }
        report_json = json.dumps(report_data, indent=2)
        report_sha256 = ProvenanceTracker.hash_data(report_json)
        report_data["report_sha256"] = report_sha256
        return json.dumps(report_data, indent=2)

    @classmethod
    def generate_csv_report(cls, sample_results: list[Any]) -> str:
        """Generate per-sample CSV export."""
        output = io.StringIO()
        fieldnames = [
            "event_id",
            "source",
            "prediction",
            "is_predicted_anomaly",
            "anomaly_score",
            "detection_threshold",
            "classification_outcome",
            "ground_truth_label",
            "is_attack_ground_truth",
            "is_held_out_unseen",
            "latency_ms",
            "policy_verdict",
            "risk_score",
            "expected_policy_outcome",
            "outcome_matched",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for s in sample_results:
            d = s.to_dict() if hasattr(s, "to_dict") else s
            row = {k: d.get(k) for k in fieldnames}
            writer.writerow(row)

        return output.getvalue()

    @classmethod
    def save_reports(
        cls,
        output_dir: str | Path,
        experiment_id: str,
        provenance: dict[str, Any],
        evaluation_results: dict[str, Any],
    ) -> tuple[Path, Path]:
        """Save both JSON and CSV evaluation reports to disk."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{experiment_id.lower()}_evaluation_report.json"
        csv_path = out_dir / f"{experiment_id.lower()}_samples.csv"

        json_text = cls.generate_json_report(experiment_id, provenance, evaluation_results)
        csv_text = cls.generate_csv_report(evaluation_results.get("sample_results", []))

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_text)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_text)

        return json_path, csv_path

