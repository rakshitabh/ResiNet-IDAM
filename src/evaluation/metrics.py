"""Scientific evaluation metrics engine for intrusion detection and LOCO benchmarking."""

from __future__ import annotations

from typing import Any


def determine_classification_result(is_predicted_anomaly: bool, is_actual_attack: bool) -> str:
    """Classify model outcome against isolated ground truth."""
    if is_predicted_anomaly and is_actual_attack:
        return "TP"
    elif not is_predicted_anomaly and not is_actual_attack:
        return "TN"
    elif is_predicted_anomaly and not is_actual_attack:
        return "FP"
    else:
        return "FN"


def calculate_classification_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Compute standard research-grade classification metrics."""
    total = tp + fp + fn + tn
    if total == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "fpr": 0.0,
            "fnr": 0.0,
            "accuracy": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / total

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "accuracy": round(accuracy, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def calculate_per_class_recall(class_breakdown: dict[str, dict[str, int]]) -> dict[str, float]:
    """Calculate recall per attack family / class."""
    per_class: dict[str, float] = {}
    for class_name, counts in class_breakdown.items():
        tp = counts.get("tp", 0)
        fn = counts.get("fn", 0)
        total_pos = tp + fn
        if total_pos > 0:
            per_class[class_name] = round(tp / total_pos, 4)
        else:
            per_class[class_name] = 0.0
    return per_class

