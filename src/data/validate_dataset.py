"""Validate the HAI feature contract and preprocessing artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = DEFAULT_ROOT / "data" / "processed" / "hai-train-scaled.csv"
DEFAULT_TEST = DEFAULT_ROOT / "data" / "processed" / "hai-test-scaled.csv"
DEFAULT_SCALER = DEFAULT_ROOT / "models" / "standard_scaler.joblib"
DEFAULT_FEATURES = DEFAULT_ROOT / "models" / "feature_columns.json"
DEFAULT_METADATA = DEFAULT_ROOT / "models" / "preprocessing_metadata.json"
DEFAULT_DISTRIBUTION = (
    DEFAULT_ROOT / "data" / "processed" / "feature_distribution_report.csv"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "reports" / "dataset-validation.json"


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def validate(
    train_path: Path,
    test_path: Path,
    scaler_path: Path,
    feature_path: Path,
    metadata_path: Path,
    distribution_path: Path,
) -> dict[str, Any]:
    train = _load(train_path)
    test = _load(test_path)
    errors: list[str] = []
    warnings: list[str] = []

    if train is None:
        errors.append(f"Training dataset not found: {train_path}")
    if test is None:
        errors.append(f"Test dataset not found: {test_path}")
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "training_data_available": train is not None,
            "test_data_available": test is not None,
        }

    assert train is not None and test is not None
    train_features = [c for c in train.columns if c not in {"timestamp", "label"}]
    test_features = [c for c in test.columns if c not in {"timestamp", "label"}]

    if train_features != test_features:
        errors.append("Training and test feature names/order do not match.")
    if "timestamp" not in train.columns or "timestamp" not in test.columns:
        errors.append("Both datasets must contain a timestamp column.")
    if "label" in train.columns:
        warnings.append(
            "Training data contains a label column; verify that training rows "
            "are restricted to trusted normal observations."
        )
    if "label" not in test.columns:
        errors.append("Test dataset must contain a label column for evaluation.")

    for name, frame in (("training", train), ("test", test)):
        values = frame[train_features].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            errors.append(f"{name.title()} features contain NaN or infinite values.")
        timestamps = pd.to_datetime(frame.get("timestamp"), errors="coerce")
        if timestamps.isna().any():
            errors.append(f"{name.title()} timestamps contain invalid values.")

    if "label" in test.columns:
        labels = set(test["label"].dropna().unique().tolist())
        if not labels.issubset({0, 1}):
            errors.append(f"Unexpected test label values: {sorted(labels)}")
        if test["label"].isna().any():
            errors.append("Test labels contain missing values.")

    scaler_info: dict[str, Any] = {
        "path": str(scaler_path),
        "exists": scaler_path.exists(),
        "feature_contract_path": str(feature_path),
        "feature_contract_exists": feature_path.exists(),
        "metadata_path": str(metadata_path),
        "metadata_exists": metadata_path.exists(),
    }
    if feature_path.exists():
        saved_features = json.loads(feature_path.read_text(encoding="utf-8"))
        if saved_features != train_features:
            errors.append("Saved feature contract does not match the dataset.")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scaler_info["metadata"] = metadata
        if metadata.get("feature_columns") != train_features:
            errors.append("Preprocessing metadata feature contract does not match.")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        scaler_info.update(
            {
                "type": type(scaler).__name__,
                "n_features_in": getattr(scaler, "n_features_in_", None),
                "feature_names_in": getattr(scaler, "feature_names_in_", None),
            }
        )
        if getattr(scaler, "n_features_in_", len(train_features)) != len(train_features):
            errors.append("Scaler feature count does not match the model feature count.")
        if hasattr(scaler, "feature_names_in_") and list(scaler.feature_names_in_) != train_features:
            errors.append("Scaler feature names/order do not match the feature contract.")
    else:
        warnings.append(
            "Scaler artifact is unavailable; training-only scaler provenance cannot "
            "be verified from this checkout."
        )

    distribution_info: dict[str, Any] = {
        "path": str(distribution_path),
        "exists": distribution_path.exists(),
    }
    if distribution_path.exists():
        distribution = pd.read_csv(distribution_path)
        expected = set(train_features)
        actual = set(distribution.get("feature", []))
        unexplained = sorted(expected - actual)
        if unexplained:
            warnings.append(
                "Feature distribution report is missing: " + ", ".join(unexplained)
            )
        distribution_info["rows"] = int(len(distribution))
        distribution_info["features_varying_only_in_test"] = (
            distribution.loc[
                (distribution["train_std"] == 0)
                & (distribution["test_std"] > 0),
                "feature",
            ].tolist()
            if {"train_std", "test_std", "feature"}.issubset(distribution.columns)
            else []
        )
    else:
        warnings.append("Feature distribution report is unavailable.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "training_data_available": True,
        "test_data_available": True,
        "feature_count": len(train_features),
        "feature_columns": train_features,
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_label_distribution": {
            str(label): int(count)
            for label, count in test["label"].value_counts(dropna=False).items()
        }
        if "label" in test.columns
        else {},
        "scaler": scaler_info,
        "distribution_report": distribution_info,
        "normal_only_training_verified": False,
        "normal_only_training_note": (
            "The official HAI documentation identifies hai-train1 as a normal "
            "dataset, but raw-file provenance and release version are not present "
            "in this checkout."
            if train is None
            else "The official HAI documentation identifies hai-train1 as a normal "
            "dataset; preserve the raw source and checksum to verify file lineage."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--distribution", type=Path, default=DEFAULT_DISTRIBUTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = validate(
        args.train,
        args.test,
        args.scaler,
        args.features,
        args.metadata,
        args.distribution,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved validation report: {args.output}")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
