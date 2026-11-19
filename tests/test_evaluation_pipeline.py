"""Comprehensive automated test suite for Research Evaluation & LOCO Pipeline (Sprint 7)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.evaluation.dataset_splitter import DatasetSplitter
from src.evaluation.experiment_config import STANDARD_EXPERIMENTS
from src.evaluation.loco_runner import LocoExperimentRunner
from src.evaluation.metrics import (
    calculate_classification_metrics,
    determine_classification_result,
)
from src.evaluation.models import (
    ClassificationOutcome,
    DetectorOutput,
    EvaluationRecord,
    InferencePayload,
)
from src.evaluation.prediction_evaluator import PredictionEvaluator
from src.evaluation.provenance import ProvenanceTracker
from src.simulation.data.telemetry_resolver import TelemetryResolver


ROOT = Path(__file__).resolve().parents[1]


def test_ground_truth_isolation():
    """Rule 4 & Phase 4: Model predictions must be completely independent of ground-truth labels."""
    output = DetectorOutput(
        source="IT",
        detector_name="IDS2025 RF",
        prediction="Dos/DDos",
        is_anomaly=True,
        anomaly_score=0.88,
        threshold=0.50,
    )

    # 1. Evaluate with malicious ground truth
    gt_malicious = EvaluationRecord(
        event_id="EVT_01",
        source="IT",
        ground_truth_label="Dos/DDos",
        is_attack_ground_truth=True,
        expected_policy_outcome="DENY",
    )
    res1 = PredictionEvaluator.evaluate_sample(output, gt_malicious, policy_verdict="DENY")
    assert res1.classification_outcome == ClassificationOutcome.TP
    assert res1.is_predicted_anomaly is True

    # 2. Mutate ground-truth label to Normal AFTER inference
    gt_mutated = EvaluationRecord(
        event_id="EVT_01",
        source="IT",
        ground_truth_label="Normal",
        is_attack_ground_truth=False,
        expected_policy_outcome="ALLOW",
    )
    res2 = PredictionEvaluator.evaluate_sample(output, gt_mutated, policy_verdict="DENY")
    
    # Prediction itself does not change, but outcome evaluation reflects the new isolated truth
    assert res2.prediction == output.prediction
    assert res2.is_predicted_anomaly == output.is_anomaly
    assert res2.classification_outcome == ClassificationOutcome.FP


def test_missing_ground_truth_not_counted_as_tn():
    """Phase 4.1: Missing ground truth must be marked UNAVAILABLE, not True Negative."""
    output = DetectorOutput(
        source="IT",
        detector_name="IDS2025 RF",
        prediction="Normal",
        is_anomaly=False,
        anomaly_score=0.10,
        threshold=0.50,
    )
    gt_missing = EvaluationRecord(
        event_id="EVT_UNKNOWN",
        source="IT",
        ground_truth_label=None,
        is_attack_ground_truth=None,
    )
    res = PredictionEvaluator.evaluate_sample(output, gt_missing)
    assert res.classification_outcome == ClassificationOutcome.UNAVAILABLE


def test_loco_dataset_splitter_leakage_safe():
    """Phase 2 & Sprint 2: Held-out attack class must be completely absent from train and calibration."""
    data_file = ROOT / "data" / "processed" / "ids2025_sample.csv"
    assert data_file.exists(), "ids2025_sample.csv not found"
    df = pd.read_csv(data_file)

    splitter = DatasetSplitter(random_seed=42)
    held_out = "Botnet ARES"
    manifest = splitter.split_loco_it(df, held_out_class=held_out)

    # 1. Held-out class must have ZERO samples in training
    assert manifest.training_class_counts.get(held_out, 0) == 0, "Held-out class leaked into training set!"
    # 2. Held-out class must have ZERO samples in calibration
    assert manifest.calibration_class_counts.get(held_out, 0) == 0, "Held-out class leaked into calibration set!"
    # 3. Calibration set must be 100% Normal samples
    for c in manifest.calibration_class_counts.keys():
        assert c.lower() in ("normal", "benign"), f"Non-normal sample found in calibration: {c}"
    # 4. Held-out class must be present in test set
    assert manifest.test_class_counts.get(held_out, 0) > 0, "Held-out class missing from test set!"


def test_split_manifest_deterministic_reproducibility(tmp_path):
    """Phase 9 & Sprint 2: Same seed and dataset must yield identical sample splits and hashes."""
    data_file = ROOT / "data" / "processed" / "ids2025_sample.csv"
    df = pd.read_csv(data_file)

    splitter1 = DatasetSplitter(random_seed=123)
    manifest1 = splitter1.split_loco_it(df, held_out_class="Infiltration")

    splitter2 = DatasetSplitter(random_seed=123)
    manifest2 = splitter2.split_loco_it(df, held_out_class="Infiltration")

    assert manifest1.training_indices == manifest2.training_indices
    assert manifest1.calibration_indices == manifest2.calibration_indices
    assert manifest1.test_indices == manifest2.test_indices

    p1 = manifest1.save(tmp_path)
    loaded = manifest1.load(p1)
    assert loaded.training_indices == manifest1.training_indices


def test_metrics_edge_case_handling():
    """Phase 5.4: Handle zero denominators and missing samples without erroneous 0.00 reporting."""
    # Test 1: No positive samples in test partition
    res_no_pos = calculate_classification_metrics(tp=0, fp=2, fn=0, tn=10)
    assert res_no_pos["recall"] is None  # Undefined, not 0.00
    assert res_no_pos["precision"] == 0.0
    assert res_no_pos["fpr"] == round(2 / 12, 4)

    # Test 2: No negative samples in test partition
    res_no_neg = calculate_classification_metrics(tp=5, fp=0, fn=1, tn=0)
    assert res_no_neg["fpr"] is None  # Undefined, not 0.00
    assert res_no_neg["recall"] == round(5 / 6, 4)

    # Test 3: Completely empty
    res_empty = calculate_classification_metrics(tp=0, fp=0, fn=0, tn=0, unavailable=5)
    assert res_empty["precision"] is None
    assert res_empty["recall"] is None
    assert res_empty["unavailable"] == 5


def test_telemetry_resolver_strict_schema():
    """Phase 6.2: TelemetryResolver must enforce ordered detector feature schemas."""
    resolver = TelemetryResolver(ROOT)

    # IT NetFlow feature schema
    it_schema = resolver.it_feature_schema()
    assert len(it_schema) == 79
    it_features = resolver.resolve("IT", 0)
    assert list(it_features.keys()) == it_schema
    assert not any(np.isnan(v) or np.isinf(v) for v in it_features.values())

    # OT HAI feature schema
    ot_schema = resolver.ot_feature_schema()
    assert len(ot_schema) == 66
    ot_features = resolver.resolve("OT", 0)
    assert list(ot_features.keys()) == ot_schema
    assert not any(np.isnan(v) or np.isinf(v) for v in ot_features.values())


def test_loco_experiment_runner_end_to_end():
    """Phase 4 & Sprint 4: Run LOCO-05 (Infiltration) and verify reports generated."""
    runner = LocoExperimentRunner(ROOT)
    cfg = STANDARD_EXPERIMENTS["LOCO-05"]
    res = runner.run_experiment(cfg, max_test_samples=25)

    assert res["experiment_id"] == "LOCO-05"
    assert "metrics" in res["results"]
    assert "provenance" in res
    assert Path(res["reports"]["json_path"]).exists()
    assert Path(res["reports"]["csv_path"]).exists()

