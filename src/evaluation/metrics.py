"""Scientific evaluation metrics engine for intrusion detection and LOCO benchmarking."""

from __future__ import annotations

from typing import Any


def determine_classification_result(
    is_predicted_anomaly: bool,
    is_actual_attack: bool | None,
) -> str:
    """Classify model outcome against isolated ground truth.
    
    If ground truth is missing/None, classify as UNAVAILABLE so it is
    never falsely counted as a True Negative.
    """
    if is_actual_attack is None:
        return "UNAVAILABLE"
    if is_predicted_anomaly and is_actual_attack:
        return "TP"
    elif not is_predicted_anomaly and not is_actual_attack:
        return "TN"
    elif is_predicted_anomaly and not is_actual_attack:
        return "FP"
    else:
        return "FN"


def calculate_classification_metrics(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    unavailable: int = 0,
) -> dict[str, Any]:
    """Compute standard research-grade classification metrics with edge-case handling."""
    total_evaluated = tp + fp + fn + tn
    total_samples = total_evaluated + unavailable

    if total_evaluated == 0:
        return {
            "precision": None,
            "recall": None,
            "f1_score": None,
            "fpr": None,
            "fnr": None,
            "accuracy": None,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "unavailable": int(unavailable),
            "total_evaluated": 0,
            "total_samples": int(total_samples),
        }

    # Precision: undefined when tp + fp == 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else (0.0 if fn > 0 else None)
    # Recall: undefined when tp + fn == 0 (no actual positives in test set)
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    
    # F1-Score: harmonic mean
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    elif precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = None

    # FPR: false positive rate (undefined when fp + tn == 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    # FNR: false negative rate
    fnr = fn / (tp + fn) if (tp + fn) > 0 else None
    # Accuracy across evaluated samples
    accuracy = (tp + tn) / total_evaluated

    return {
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1_score": round(f1, 4) if f1 is not None else None,
        "fpr": round(fpr, 4) if fpr is not None else None,
        "fnr": round(fnr, 4) if fnr is not None else None,
        "accuracy": round(accuracy, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "unavailable": int(unavailable),
        "total_evaluated": int(total_evaluated),
        "total_samples": int(total_samples),
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

