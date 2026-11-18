"""Confusion matrix and classification breakdown tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_per_class_recall,
    determine_classification_result,
)


@dataclass
class ConfusionMatrixTracker:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    class_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def record_prediction(
        self,
        is_predicted_anomaly: bool,
        is_actual_attack: bool,
        attack_class: str = "Attack",
    ) -> str:
        res = determine_classification_result(is_predicted_anomaly, is_actual_attack)
        if res == "TP":
            self.tp += 1
            if is_actual_attack:
                self._increment_class(attack_class, "tp")
        elif res == "FP":
            self.fp += 1
        elif res == "FN":
            self.fn += 1
            if is_actual_attack:
                self._increment_class(attack_class, "fn")
        elif res == "TN":
            self.tn += 1

        return res

    def _increment_class(self, attack_class: str, metric: str) -> None:
        if attack_class not in self.class_counts:
            self.class_counts[attack_class] = {"tp": 0, "fn": 0}
        self.class_counts[attack_class][metric] = self.class_counts[attack_class].get(metric, 0) + 1

    def metrics(self) -> dict[str, float]:
        return calculate_classification_metrics(self.tp, self.fp, self.fn, self.tn)

    def per_class_recall(self) -> dict[str, float]:
        return calculate_per_class_recall(self.class_counts)

    def to_matrix_dict(self) -> dict[str, Any]:
        return {
            "predicted_attack_actual_attack": self.tp,
            "predicted_normal_actual_attack": self.fn,
            "predicted_attack_actual_normal": self.fp,
            "predicted_normal_actual_normal": self.tn,
            "metrics": self.metrics(),
            "per_class_recall": self.per_class_recall(),
        }

