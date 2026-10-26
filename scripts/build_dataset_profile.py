"""Build a compact, explainable dataset profile for the API and dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "hai-train-scaled.csv"
TEST_PATH = ROOT / "data" / "processed" / "hai-test-scaled.csv"
DISTRIBUTION_PATH = ROOT / "data" / "processed" / "feature_distribution_report.csv"
OUTPUT_PATH = ROOT / "reports" / "dataset-profile.json"


PROCESS_NAMES = {
    "P1": "Boiler process",
    "P2": "Turbine process",
    "P3": "Water-treatment process",
    "P4": "HIL simulation / power process",
}


def describe_feature(name: str) -> tuple[str, str, str]:
    process_match = re.match(r"(P[1-4])_", name)
    process = PROCESS_NAMES.get(process_match.group(1), "Control-system auxiliary signal") if process_match else "Control-system auxiliary signal"
    upper = name.upper()
    if "TIT" in upper or "TT" in upper:
        signal_type, meaning = "Temperature", "Temperature measurement or temperature-related process signal."
    elif "PIT" in upper or "PT" in upper:
        signal_type, meaning = "Pressure", "Pressure measurement or pressure-related process signal."
    elif "FIT" in upper or "FT" in upper:
        signal_type, meaning = "Flow", "Flow measurement or flow-related process signal."
    elif "LIT" in upper or "LCV" in upper or "LC" in upper:
        signal_type, meaning = "Level / valve", "Liquid level measurement or level-control valve signal."
    elif "VIB" in upper or upper.startswith("P2_VT"):
        signal_type, meaning = "Vibration", "Rotating-equipment vibration or vibration-related signal."
    elif "FCV" in upper or "PCV" in upper:
        signal_type, meaning = "Control valve", "Control-valve demand, position, or controller signal."
    elif "PP" in upper or "PUMP" in upper:
        signal_type, meaning = "Pump / actuator", "Pump, actuator, setpoint, or actuator-control signal."
    elif any(token in upper for token in ("AUTO", "MANUAL", "MASW", "SCO", "SCST", "LAMP", "24V", "GOV")):
        signal_type, meaning = "State / control", "Discrete state, mode, switch, governor, or control-status signal."
    elif "SETPOINT" in upper or "ASSIGN" in upper or "SUM" in upper:
        signal_type, meaning = "Auxiliary control", "Controller setpoint, assignment, or derived control signal."
    else:
        signal_type, meaning = "Process signal", "Numeric industrial telemetry signal; exact semantics require HAI tag documentation."
    return process, signal_type, meaning


def main() -> None:
    train = pd.read_csv(TRAIN_PATH, nrows=2)
    test = pd.read_csv(TEST_PATH, nrows=2)
    distribution = pd.read_csv(DISTRIBUTION_PATH)
    feature_columns = [column for column in train.columns if column not in {"timestamp", "label"}]
    features = []
    distribution_by_name = distribution.set_index("feature").to_dict("index")
    for name in feature_columns:
        process, signal_type, meaning = describe_feature(name)
        stats = distribution_by_name.get(name, {})
        features.append(
            {
                "name": name,
                "process": process,
                "signal_type": signal_type,
                "meaning": meaning,
                "model_usage": "Numeric Autoencoder input after training-fitted scaling.",
                "train_mean": stats.get("train_mean"),
                "train_std": stats.get("train_std"),
                "test_mean": stats.get("test_mean"),
                "test_std": stats.get("test_std"),
                "mean_difference": stats.get("mean_difference"),
            }
        )

    profile = {
        "dataset": "HAI",
        "source": "https://github.com/icsdataset/hai",
        "training": {"file": str(TRAIN_PATH.relative_to(ROOT)), "rows": 280800, "columns": len(train.columns)},
        "test": {"file": str(TEST_PATH.relative_to(ROOT)), "rows": 54000, "columns": len(test.columns), "normal_rows": 51019, "attack_rows": 2981},
        "model_input": {"feature_count": len(feature_columns), "excluded_columns": ["timestamp", "label"], "sampling_interval_seconds": 1},
        "measurements": [
            {"name": "Reconstruction error", "unit": "MSE", "meaning": "Mean squared difference between the original feature vector and Autoencoder reconstruction."},
            {"name": "Anomaly decision", "unit": "boolean", "meaning": "True when reconstruction error exceeds the validation-selected threshold."},
            {"name": "Anomaly severity", "unit": "0-1", "meaning": "Normalized detector severity used as one input to contextual risk scoring."},
            {"name": "Risk score", "unit": "0-100", "meaning": "Weighted contextual score combining anomaly, behavior, simulated trust, and asset criticality."},
        ],
        "label_usage": "Ground-truth label is reserved for offline evaluation and is never accepted by the inference endpoint.",
        "feature_groups": [
            {"name": process, "count": sum(feature["process"] == process for feature in features)}
            for process in sorted({feature["process"] for feature in features})
        ],
        "features": features,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Saved dataset profile: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
