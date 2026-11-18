"""Machine-readable JSON report generator with cryptographic audit hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from src.simulation.models.simulation_run import SimulationRun


class JsonReportGenerator:
    """Generates a complete, verifiable JSON report of a simulation execution run."""

    @classmethod
    def generate(cls, run: SimulationRun) -> str:
        data = run.to_dict()
        # Compute execution output hash
        raw_dump = json.dumps(data, sort_keys=True)
        report_hash = hashlib.sha256(raw_dump.encode("utf-8")).hexdigest()
        data["audit_metadata"] = {
            "output_report_sha256": report_hash,
            "dataset_sha256": run.dataset_hash,
            "safety_boundary": "SIMULATED_DECISION_SUPPORT_ONLY_NO_PHYSICAL_ACTUATION",
        }
        return json.dumps(data, indent=2)

    @classmethod
    def save(cls, run: SimulationRun, target_path: Path) -> Path:
        content = cls.generate(run)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        return target_path

