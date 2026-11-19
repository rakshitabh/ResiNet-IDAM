"""Dataset split manifest model and persistence engine (Sprint 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.evaluation.provenance import ProvenanceTracker


@dataclass
class SplitManifest:
    """Records the exact sample IDs, class distributions, and hashes for an experiment split."""
    manifest_id: str
    experiment_id: str
    dataset_name: str
    domain: str  # 'IT' or 'OT'
    held_out_attack_class: str | None
    random_seed: int
    created_at_iso: str
    
    # Sample index lists
    training_indices: list[int]
    calibration_indices: list[int]
    test_indices: list[int]

    # Breakdown counts
    training_class_counts: dict[str, int]
    calibration_class_counts: dict[str, int]
    test_class_counts: dict[str, int]

    # Data checksums
    dataset_hash: str
    manifest_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "experiment_id": self.experiment_id,
            "dataset_name": self.dataset_name,
            "domain": self.domain,
            "held_out_attack_class": self.held_out_attack_class,
            "random_seed": self.random_seed,
            "created_at_iso": self.created_at_iso,
            "sample_counts": {
                "training": len(self.training_indices),
                "calibration": len(self.calibration_indices),
                "test": len(self.test_indices),
                "total": len(self.training_indices) + len(self.calibration_indices) + len(self.test_indices),
            },
            "training_indices": self.training_indices,
            "calibration_indices": self.calibration_indices,
            "test_indices": self.test_indices,
            "training_class_counts": self.training_class_counts,
            "calibration_class_counts": self.calibration_class_counts,
            "test_class_counts": self.test_class_counts,
            "dataset_hash": self.dataset_hash,
            "manifest_hash": self.manifest_hash,
            "metadata": self.metadata,
        }

    def save(self, output_dir: str | Path) -> Path:
        """Persist manifest to JSON file."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"{self.manifest_id}.json"
        
        # Compute self-hash excluding manifest_hash
        data_dict = self.to_dict()
        data_dict["manifest_hash"] = ""
        self.manifest_hash = ProvenanceTracker.hash_data(json.dumps(data_dict, sort_keys=True))
        data_dict["manifest_hash"] = self.manifest_hash

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2)
        return file_path

    @classmethod
    def load(cls, file_path: str | Path) -> SplitManifest:
        """Load manifest from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            manifest_id=data["manifest_id"],
            experiment_id=data["experiment_id"],
            dataset_name=data["dataset_name"],
            domain=data["domain"],
            held_out_attack_class=data["held_out_attack_class"],
            random_seed=data["random_seed"],
            created_at_iso=data["created_at_iso"],
            training_indices=data["training_indices"],
            calibration_indices=data["calibration_indices"],
            test_indices=data["test_indices"],
            training_class_counts=data["training_class_counts"],
            calibration_class_counts=data["calibration_class_counts"],
            test_class_counts=data["test_class_counts"],
            dataset_hash=data["dataset_hash"],
            manifest_hash=data.get("manifest_hash", ""),
            metadata=data.get("metadata", {}),
        )

