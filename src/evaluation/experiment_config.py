"""Experiment configuration models and presets for LOCO and baseline benchmarks (Sprint 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExperimentConfig:
    """Defines full parameters for an intrusion detection or LOCO experiment."""
    experiment_id: str
    domain: str  # 'IT', 'OT', or 'IT + OT'
    dataset: str  # 'ids2025_sample.csv', 'hai-test-scaled.csv', etc.
    held_out_attack: Optional[str] = None
    detector_type: str = "Existing RF / AE"
    threshold_policy: str = "Frozen normal-validation threshold"
    random_seed: int = 42
    description: str = ""
    run_status: str = "READY"  # 'READY', 'RUNNING', 'COMPLETED'
    created_at_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "domain": self.domain,
            "dataset": self.dataset,
            "held_out_attack": self.held_out_attack,
            "detector_type": self.detector_type,
            "threshold_policy": self.threshold_policy,
            "random_seed": self.random_seed,
            "description": self.description,
            "run_status": self.run_status,
            "created_at_iso": self.created_at_iso,
            "metadata": self.metadata,
        }

    def save(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: str | Path) -> ExperimentConfig:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# Standard LOCO and baseline experiment configurations matching Phase 2.1 & Phase 11
STANDARD_EXPERIMENTS: dict[str, ExperimentConfig] = {
    "LOCO-04": ExperimentConfig(
        experiment_id="LOCO-04",
        domain="IT",
        dataset="ids2025_sample.csv",
        held_out_attack="Botnet ARES",
        detector_type="IDS2025 RF / NetFlow AE",
        threshold_policy="Frozen normal-validation threshold (tau=0.6383)",
        random_seed=42,
        description="Leave-One-Class-Out evaluation: Botnet ARES held out completely from training/calibration.",
    ),
    "LOCO-05": ExperimentConfig(
        experiment_id="LOCO-05",
        domain="IT",
        dataset="ids2025_sample.csv",
        held_out_attack="Infiltration",
        detector_type="IDS2025 RF / NetFlow AE",
        threshold_policy="Frozen normal-validation threshold (tau=0.6383)",
        random_seed=42,
        description="Leave-One-Class-Out evaluation: Infiltration held out completely from training/calibration.",
    ),
    "LOCO-01": ExperimentConfig(
        experiment_id="LOCO-01",
        domain="IT",
        dataset="ids2025_sample.csv",
        held_out_attack="Dos/DDos",
        detector_type="IDS2025 RF / NetFlow AE",
        threshold_policy="Frozen normal-validation threshold (tau=0.6383)",
        random_seed=42,
        description="Leave-One-Class-Out evaluation: DoS/DDoS held out completely from training/calibration.",
    ),
    "LOCO-02": ExperimentConfig(
        experiment_id="LOCO-02",
        domain="IT",
        dataset="ids2025_sample.csv",
        held_out_attack="PortScan",
        detector_type="IDS2025 RF / NetFlow AE",
        threshold_policy="Frozen normal-validation threshold (tau=0.6383)",
        random_seed=42,
        description="Leave-One-Class-Out evaluation: PortScan held out completely from training/calibration.",
    ),
    "LOCO-03": ExperimentConfig(
        experiment_id="LOCO-03",
        domain="IT",
        dataset="ids2025_sample.csv",
        held_out_attack="Brute Force",
        detector_type="IDS2025 RF / NetFlow AE",
        threshold_policy="Frozen normal-validation threshold (tau=0.6383)",
        random_seed=42,
        description="Leave-One-Class-Out evaluation: Brute Force held out completely from training/calibration.",
    ),
    "OT-01": ExperimentConfig(
        experiment_id="OT-01",
        domain="OT",
        dataset="hai-test-scaled.csv",
        held_out_attack="Actuator Tamper",
        detector_type="HAI Autoencoder (66 features)",
        threshold_policy="Frozen normal-validation threshold (tau=0.8219)",
        random_seed=42,
        description="OT Industrial evaluation: Physical process valve and pump tampering episodes.",
    ),
    "FULL-01": ExperimentConfig(
        experiment_id="FULL-01",
        domain="IT + OT",
        dataset="zero_trust_simulation_scenarios.csv",
        held_out_attack="Multi-Stage Kill-Chain",
        detector_type="IT + OT Detectors + Zero-Trust Engine",
        threshold_policy="Domain-specific frozen thresholds",
        random_seed=42,
        description="12-event full cyber-physical kill-chain progression across IT and OT domains.",
    ),
}

