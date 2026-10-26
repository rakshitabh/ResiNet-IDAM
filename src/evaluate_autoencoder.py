"""Evaluate the saved HAI Autoencoder without using labels for inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "hai-test-scaled.csv"
MODEL_PATH = ROOT / "models" / "autoencoder.keras"
THRESHOLD_PATH = ROOT / "models" / "anomaly_threshold.json"
METRICS_PATH = ROOT / "reports" / "baseline_metrics.json"
SCORES_PATH = ROOT / "reports" / "test_anomaly_scores.csv"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold artifact not found: {THRESHOLD_PATH}")

    frame = pd.read_csv(DATA_PATH)
    metadata = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    threshold = float(metadata["threshold"])

    if "label" not in frame.columns:
        raise ValueError("The evaluation dataset must contain a label column.")
    actual_features = [
        column for column in frame.columns if column not in {"timestamp", "label"}
    ]
    if actual_features != feature_columns:
        raise ValueError("Evaluation feature names/order do not match the artifact.")

    X = frame[feature_columns].to_numpy(dtype=np.float32)
    y_true = frame["label"].to_numpy(dtype=np.int32)
    if not np.isfinite(X).all():
        raise ValueError("Evaluation features contain NaN or infinite values.")

    model = keras.models.load_model(MODEL_PATH)
    reconstructed = model.predict(X, batch_size=512, verbose=0)
    errors = np.mean(np.square(X - reconstructed), axis=1)
    y_pred = (errors > threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "model": metadata.get("model_version", "dense-autoencoder-baseline-v1"),
        "threshold": threshold,
        "threshold_selection": metadata["threshold_selection"],
        "test_rows": int(len(y_true)),
        "normal_rows": int((y_true == 0).sum()),
        "attack_rows": int((y_true == 1).sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "roc_auc": float(roc_auc_score(y_true, errors)),
        "pr_auc": float(average_precision_score(y_true, errors)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = frame[["timestamp", "label"]].copy()
    output["reconstruction_error"] = errors
    output["predicted_anomaly"] = y_pred
    output.to_csv(SCORES_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Scores saved: {SCORES_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
