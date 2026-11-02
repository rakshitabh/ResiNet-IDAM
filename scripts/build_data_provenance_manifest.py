"""Build a comprehensive, hash-verified data provenance manifest for ResiNet-IDAM v2.0."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def get_sha256(path: Path) -> str:
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    test_lbl_path = ROOT / "data" / "processed" / "hai-test-labeled.csv"
    test_lbl = pd.read_csv(test_lbl_path)
    labels = test_lbl["label"].values
    timestamps = test_lbl["timestamp"].values

    attack_indices = np.where(labels == 1)[0]
    diffs = np.diff(attack_indices)
    split_points = np.where(diffs > 1)[0]

    windows = []
    starts = [attack_indices[0]] + [attack_indices[i + 1] for i in split_points]
    ends = [attack_indices[i] for i in split_points] + [attack_indices[-1]]

    for s, e in zip(starts, ends):
        windows.append(
            {
                "start_row_index": int(s),
                "end_row_index": int(e),
                "duration_seconds": int(e - s + 1),
                "start_timestamp": str(timestamps[s]),
                "end_timestamp": str(timestamps[e]),
            }
        )

    manifest = {
        "schema_version": "2.0.0",
        "created_at": "2026-11-01T18:00:00Z",
        "project": "ResiNet-IDAM v2.0",
        "description": "Reproducible Data Provenance, Integrity Hashes, and Attack Label Manifest",
        "datasets": {
            "OT_telemetry": {
                "benchmark_name": "HAI (Hardware-in-the-Loop Anomaly Detection) 21.03/22.04",
                "subsystems": [
                    "P1: Boiler / Water Treatment",
                    "P2: Turbine / Generator",
                    "P3: Water Treatment / RO",
                    "P4: Smart Power / Thermal",
                ],
                "feature_count": 66,
                "sampling_rate": "1 sample / second (1 Hz)",
                "training_partition": {
                    "file": "data/processed/hai-train-scaled.csv",
                    "sha256": get_sha256(
                        ROOT / "data" / "processed" / "hai-train-scaled.csv"
                    ),
                    "rows": 280800,
                    "time_span": {
                        "start": "2022-08-04 18:00:01",
                        "end": "2022-08-08 00:00:00",
                    },
                    "label_type": "Unlabeled normal operational baseline",
                },
                "test_partition": {
                    "file_scaled": "data/processed/hai-test-scaled.csv",
                    "file_unscaled": "data/processed/hai-test-labeled.csv",
                    "sha256_scaled": get_sha256(
                        ROOT / "data" / "processed" / "hai-test-scaled.csv"
                    ),
                    "sha256_unscaled": get_sha256(
                        ROOT / "data" / "processed" / "hai-test-labeled.csv"
                    ),
                    "rows": 54000,
                    "time_span": {
                        "start": "2022-08-12 16:00:01",
                        "end": "2022-08-13 07:00:00",
                    },
                    "normal_samples": int((labels == 0).sum()),
                    "attack_samples": int((labels == 1).sum()),
                    "attack_percentage": float((labels == 1).mean() * 100),
                    "total_attack_windows": len(windows),
                    "first_attack_row": int(windows[0]["start_row_index"]),
                    "first_attack_timestamp": windows[0]["start_timestamp"],
                    "attack_windows": windows,
                },
                "scaler_artifact": {
                    "file": "models/standard_scaler.joblib",
                    "sha256": get_sha256(ROOT / "models" / "standard_scaler.joblib"),
                    "feature_contract_file": "models/feature_columns.json",
                    "feature_contract_sha256": get_sha256(
                        ROOT / "models" / "feature_columns.json"
                    ),
                },
            },
            "IT_telemetry": {
                "benchmark_name": "IDS2025 Realistic Network Flow Partition",
                "feature_count_config_a": 79,
                "feature_count_config_b": 77,
                "training_partition": {
                    "file": "data/processed/it_realistic/it_train_all.npz",
                    "sha256": get_sha256(
                        ROOT / "data" / "processed" / "it_realistic" / "it_train_all.npz"
                    ),
                    "rows": 73375,
                },
                "test_partition": {
                    "file": "data/processed/it_realistic/it_test_all.npz",
                    "sha256": get_sha256(
                        ROOT / "data" / "processed" / "it_realistic" / "it_test_all.npz"
                    ),
                    "rows": 18344,
                    "classes": {
                        "Normal": 5237,
                        "Dos/DDos": 5191,
                        "PortScan": 5082,
                        "Brute Force": 2040,
                        "Web Attack": 413,
                        "Botnet ARES": 375,
                        "Infiltration": 6,
                    },
                },
                "preprocessing_pipeline": {
                    "file": "models/it_realistic/preprocessing_all.joblib",
                    "sha256": get_sha256(
                        ROOT
                        / "models"
                        / "it_realistic"
                        / "preprocessing_all.joblib"
                    ),
                    "metadata_file": "data/processed/it_realistic/metadata.json",
                },
            },
        },
    }

    out_path = ROOT / "reports" / "data_provenance_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Data provenance manifest written: {out_path}")


if __name__ == "__main__":
    main()

