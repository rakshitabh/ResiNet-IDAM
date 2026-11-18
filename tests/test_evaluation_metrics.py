"""Unit tests for the research evaluation metrics and confusion matrix tracker."""

import pytest
from src.evaluation.confusion_matrix import ConfusionMatrixTracker
from src.evaluation.distribution_tracker import DistributionTracker
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_per_class_recall,
    determine_classification_result,
)


def test_determine_classification_result():
    assert determine_classification_result(is_predicted_anomaly=True, is_actual_attack=True) == "TP"
    assert determine_classification_result(is_predicted_anomaly=False, is_actual_attack=False) == "TN"
    assert determine_classification_result(is_predicted_anomaly=True, is_actual_attack=False) == "FP"
    assert determine_classification_result(is_predicted_anomaly=False, is_actual_attack=True) == "FN"


def test_calculate_classification_metrics():
    # 472 TP, 18 FP, 28 FN, 948 TN (matching image counts)
    metrics = calculate_classification_metrics(tp=472, fp=18, fn=28, tn=948)
    # Precision = 472 / (472 + 18) = 472 / 490 = ~0.9633 -> 0.96
    assert metrics["precision"] == pytest.approx(0.9633, rel=1e-3)
    # Recall = 472 / (472 + 28) = 472 / 500 = 0.9440 -> 0.94
    assert metrics["recall"] == pytest.approx(0.944, rel=1e-3)
    # FPR = 18 / (18 + 948) = 18 / 966 = ~0.0186 -> 0.02
    assert metrics["fpr"] == pytest.approx(0.0186, rel=1e-2)
    assert metrics["tp"] == 472
    assert metrics["fp"] == 18


def test_confusion_matrix_tracker():
    tracker = ConfusionMatrixTracker()
    tracker.record_prediction(is_predicted_anomaly=True, is_actual_attack=True, attack_class="Botnet ARES")
    tracker.record_prediction(is_predicted_anomaly=False, is_actual_attack=True, attack_class="Botnet ARES")
    tracker.record_prediction(is_predicted_anomaly=True, is_actual_attack=False)
    tracker.record_prediction(is_predicted_anomaly=False, is_actual_attack=False)

    m = tracker.metrics()
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["fp"] == 1
    assert m["tn"] == 1

    recalls = tracker.per_class_recall()
    assert recalls["Botnet ARES"] == 0.5


def test_distribution_tracker():
    dist = DistributionTracker()
    dist.record_score(0.12, is_attack=False)
    dist.record_score(0.85, is_attack=True)
    data = dist.density_data(bins=10)
    assert len(data["bin_centers"]) == 10
    assert len(data["normal_density"]) == 10
    assert len(data["attack_density"]) == 10

