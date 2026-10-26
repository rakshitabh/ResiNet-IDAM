"""Measure local baseline training, inference throughput, and process memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
from tensorflow import keras


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "hai-test-scaled.csv"
TRAIN_SCRIPT = ROOT / "src" / "train_autoencoder.py"
MODEL_PATH = ROOT / "models" / "autoencoder.keras"
OUTPUT_PATH = ROOT / "reports" / "performance-benchmark.json"


def measure_inference(batch_size: int, rows: int) -> dict[str, float | int]:
    frame = pd.read_csv(DATA_PATH, nrows=rows)
    features = [column for column in frame.columns if column not in {"timestamp", "label"}]
    values = frame[features].to_numpy(dtype=np.float32)
    process = psutil.Process()
    before = process.memory_info().rss
    model = keras.models.load_model(MODEL_PATH)
    loaded = process.memory_info().rss
    started = time.perf_counter()
    model.predict(values, batch_size=batch_size, verbose=0)
    elapsed = time.perf_counter() - started
    after = process.memory_info().rss
    return {
        "rows": len(values),
        "features": len(features),
        "batch_size": batch_size,
        "elapsed_seconds": round(elapsed, 4),
        "rows_per_second": round(len(values) / elapsed, 2),
        "inference_latency_ms_per_row": round(elapsed * 1000 / len(values), 4),
        "rss_before_model_mb": round(before / 1024**2, 2),
        "rss_after_model_load_mb": round(loaded / 1024**2, 2),
        "rss_after_inference_mb": round(after / 1024**2, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--rows", type=int, default=54000)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    training: dict[str, float | int | str]
    if args.skip_training:
        training = {"skipped": True}
    else:
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=ROOT,
            check=False,
        )
        training = {
            "skipped": False,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "exit_code": completed.returncode,
        }
        if completed.returncode:
            raise SystemExit(completed.returncode)

    inference = measure_inference(args.batch_size, args.rows)
    report = {
        "environment": {
            "python": sys.version,
            "tensorflow": __import__("tensorflow").__version__,
            "platform": sys.platform,
        },
        "training": training,
        "inference": inference,
        "notes": [
            "Measurements are local laptop observations, not performance guarantees.",
            "RSS includes Python, TensorFlow, model, and loaded input data.",
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved benchmark: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
