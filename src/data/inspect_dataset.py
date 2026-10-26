"""Inspect prepared HAI datasets and write a machine-readable audit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = DEFAULT_ROOT / "data" / "processed" / "hai-train-scaled.csv"
DEFAULT_TEST = DEFAULT_ROOT / "data" / "processed" / "hai-test-scaled.csv"
DEFAULT_OUTPUT = DEFAULT_ROOT / "reports" / "dataset-inspection.json"


def _timestamp_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if "timestamp" not in frame.columns:
        return {"present": False}

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    intervals = timestamps.sort_values().diff().dropna().dt.total_seconds()
    return {
        "present": True,
        "invalid": int(timestamps.isna().sum()),
        "duplicate": int(timestamps.duplicated().sum()),
        "first": timestamps.min().isoformat() if timestamps.notna().any() else None,
        "last": timestamps.max().isoformat() if timestamps.notna().any() else None,
        "interval_seconds": {
            "min": float(intervals.min()) if not intervals.empty else None,
            "median": float(intervals.median()) if not intervals.empty else None,
            "max": float(intervals.max()) if not intervals.empty else None,
            "unique": sorted(float(value) for value in intervals.unique())[:100],
        },
    }


def inspect_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}

    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include="number")
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": frame.columns.tolist(),
        "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "missing_cells": int(frame.isna().sum().sum()),
        "missing_by_column": {
            name: int(count)
            for name, count in frame.isna().sum().items()
            if count
        },
        "duplicate_rows": int(frame.duplicated().sum()),
        "constant_columns": [
            name for name in frame.columns if frame[name].nunique(dropna=False) <= 1
        ],
        "timestamp": _timestamp_summary(frame),
        "numeric_summary": json.loads(
            numeric.describe().transpose().to_json()
        )
        if not numeric.empty
        else {},
    }
    if "label" in frame.columns:
        summary["label_distribution"] = {
            str(label): int(count)
            for label, count in frame["label"].value_counts(dropna=False).items()
        }
    return summary


def build_report(train_path: Path, test_path: Path) -> dict[str, Any]:
    return {
        "tool": "src/data/inspect_dataset.py",
        "dataset": "HAI",
        "files": {
            "training": inspect_file(train_path),
            "test": inspect_file(test_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(args.train, args.test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved inspection report: {args.output}")


if __name__ == "__main__":
    main()
