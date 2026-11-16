"""Unified reporting facade for simulation artifacts."""

from __future__ import annotations

from pathlib import Path
from src.simulation.models.simulation_run import SimulationRun
from src.simulation.reporting.csv_report import CsvReportGenerator
from src.simulation.reporting.json_report import JsonReportGenerator


class SimulationReporter:
    """Orchestrates CSV and JSON report generation and persistence."""

    def __init__(self, reports_dir: str | Path) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self, run: SimulationRun) -> dict[str, Path]:
        csv_path = self.reports_dir / f"{run.run_id}_report.csv"
        json_path = self.reports_dir / f"{run.run_id}_report.json"

        CsvReportGenerator.save(run, csv_path)
        JsonReportGenerator.save(run, json_path)

        return {
            "csv": csv_path,
            "json": json_path,
        }

