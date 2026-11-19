"""Provenance and reproducibility tracker for experiments, models, and datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class ProvenanceTracker:
    """Computes cryptographic hashes and metadata for scientific auditability."""

    @staticmethod
    def hash_file(file_path: str | Path) -> str:
        """Compute SHA-256 hash of a file on disk."""
        path = Path(file_path)
        if not path.exists():
            return "FILE_NOT_FOUND"
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def hash_data(data: bytes | str) -> str:
        """Compute SHA-256 hash of in-memory data."""
        raw = data.encode("utf-8") if isinstance(data, str) else data
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def get_experiment_provenance(
        cls,
        root: str | Path,
        dataset_name: str,
        model_name: str,
        feature_schema_version: str,
        threshold: float,
        held_out_class: str | None = None,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Collect full cryptographic provenance manifest for an experiment run."""
        root_path = Path(root)
        models_dir = root_path / "models"
        data_dir = root_path / "data"

        # Model hash resolution
        model_hash = "N/A"
        if "it" in model_name.lower() or "random_forest" in model_name.lower():
            target_file = models_dir / "it_realistic" / "random_forest_all.joblib"
            model_hash = cls.hash_file(target_file)
        elif "hai" in model_name.lower() or "ot" in model_name.lower():
            target_file = models_dir / "autoencoder.keras"
            if not target_file.exists():
                target_file = models_dir / "ot_autoencoder_weights.npz"
            model_hash = cls.hash_file(target_file)

        # Dataset hash resolution
        dataset_hash = "N/A"
        if "ids2025" in dataset_name.lower():
            target_data = data_dir / "processed" / "ids2025_sample.csv"
            dataset_hash = cls.hash_file(target_data)
        elif "hai" in dataset_name.lower():
            target_data = data_dir / "processed" / "hai-test-scaled.csv"
            dataset_hash = cls.hash_file(target_data)

        return {
            "dataset_name": dataset_name,
            "dataset_hash": dataset_hash,
            "model_name": model_name,
            "model_hash": model_hash,
            "feature_schema_version": feature_schema_version,
            "threshold": float(threshold),
            "held_out_class": held_out_class,
            "random_seed": int(random_seed),
            "provenance_standard": "ResiNet-IDAM-v2.1-Reproducibility",
        }

