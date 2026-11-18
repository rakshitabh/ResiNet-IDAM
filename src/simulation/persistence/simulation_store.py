"""Isolated simulation persistence store ensuring zero production data contamination."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from src.simulation.models.simulation_run import SimulationRun


class SimulationStore:
    """Manages dedicated simulation artifacts and run history independently of production databases."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        self.sim_dir = self.root / "data" / "simulation"
        self.runs_dir = self.sim_dir / "runs"
        self.reports_dir = self.sim_dir / "reports"
        self.uploads_dir = self.sim_dir / "uploads"
        self.scenarios_dir = self.sim_dir / "scenarios"

        for d in (self.runs_dir, self.reports_dir, self.uploads_dir, self.scenarios_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save_run(self, run: SimulationRun) -> Path:
        path = self.runs_dir / f"{run.run_id}.json"
        data = run.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for path in sorted(self.runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    runs.append({
                        "run_id": data.get("run_id"),
                        "dataset_name": data.get("dataset_name"),
                        "state": data.get("state"),
                        "total_events": data.get("total_events"),
                        "processed_events": data.get("processed_events"),
                        "start_time_iso": data.get("start_time_iso"),
                        "end_time_iso": data.get("end_time_iso"),
                    })
            except Exception:
                continue
        return runs

    def save_upload(self, filename: str, content: bytes) -> Path:
        sanitized = Path(filename).name
        target = self.uploads_dir / sanitized
        with open(target, "wb") as f:
            f.write(content)
        return target

