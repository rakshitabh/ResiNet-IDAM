"""Evaluates model predictions against strictly isolated ground truth (Sprint 4)."""

from __future__ import annotations

from typing import Any
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_per_class_recall,
    determine_classification_result,
)
from src.evaluation.models import (
    ClassificationOutcome,
    DetectorOutput,
    EvaluationRecord,
    EvaluationSampleResult,
)


class PredictionEvaluator:
    """Joins detector predictions with ground-truth records post-inference and computes scientific metrics."""

    @staticmethod
    def evaluate_sample(
        detector_output: DetectorOutput,
        evaluation_record: EvaluationRecord,
        policy_verdict: str | None = None,
        risk_score: float | None = None,
    ) -> EvaluationSampleResult:
        """Evaluate a single prediction against its isolated ground truth."""
        raw_outcome = determine_classification_result(
            is_predicted_anomaly=detector_output.is_anomaly,
            is_actual_attack=evaluation_record.is_attack_ground_truth,
        )
        outcome = ClassificationOutcome(raw_outcome)

        # Policy expectation match (independent of attack classification)
        outcome_matched = None
        if evaluation_record.expected_policy_outcome and policy_verdict:
            clean_expected = evaluation_record.expected_policy_outcome.strip().upper().replace(" ", "_")
            clean_verdict = policy_verdict.strip().upper().replace(" ", "_")
            outcome_matched = (clean_expected == clean_verdict)

        return EvaluationSampleResult(
            event_id=evaluation_record.event_id,
            source=detector_output.source,
            prediction=detector_output.prediction,
            is_predicted_anomaly=detector_output.is_anomaly,
            anomaly_score=detector_output.anomaly_score,
            detection_threshold=detector_output.threshold,
            classification_outcome=outcome,
            ground_truth_label=evaluation_record.ground_truth_label,
            is_attack_ground_truth=evaluation_record.is_attack_ground_truth,
            is_held_out_unseen=evaluation_record.is_held_out_unseen,
            latency_ms=detector_output.latency_ms,
            policy_verdict=policy_verdict,
            risk_score=risk_score,
            expected_policy_outcome=evaluation_record.expected_policy_outcome,
            outcome_matched=outcome_matched,
            metadata={
                "detector_name": detector_output.detector_name,
                "model_version": detector_output.model_version,
                "confidence": detector_output.confidence,
            },
        )

    @classmethod
    def evaluate_batch(
        cls,
        predictions: list[DetectorOutput],
        ground_truths: list[EvaluationRecord],
        policy_verdicts: list[str | None] | None = None,
        risk_scores: list[float | None] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a batch of predictions against ground truths."""
        if len(predictions) != len(ground_truths):
            raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths")

        verdicts = policy_verdicts or [None] * len(predictions)
        risks = risk_scores or [None] * len(predictions)

        sample_results: list[EvaluationSampleResult] = []
        tp = fp = fn = tn = unavailable = 0
        class_breakdown: dict[str, dict[str, int]] = {}
        unseen_breakdown: dict[str, dict[str, int]] = {}

        for pred, gt, verd, r_score in zip(predictions, ground_truths, verdicts, risks):
            res = cls.evaluate_sample(pred, gt, policy_verdict=verd, risk_score=r_score)
            sample_results.append(res)

            if res.classification_outcome == ClassificationOutcome.TP:
                tp += 1
            elif res.classification_outcome == ClassificationOutcome.FP:
                fp += 1
            elif res.classification_outcome == ClassificationOutcome.FN:
                fn += 1
            elif res.classification_outcome == ClassificationOutcome.TN:
                tn += 1
            else:
                unavailable += 1

            # Track per-class recall
            if res.is_attack_ground_truth:
                cname = res.ground_truth_label or "Attack"
                if cname not in class_breakdown:
                    class_breakdown[cname] = {"tp": 0, "fn": 0}
                if res.classification_outcome == ClassificationOutcome.TP:
                    class_breakdown[cname]["tp"] += 1
                elif res.classification_outcome == ClassificationOutcome.FN:
                    class_breakdown[cname]["fn"] += 1

                if res.is_held_out_unseen:
                    if cname not in unseen_breakdown:
                        unseen_breakdown[cname] = {"tp": 0, "fn": 0}
                    if res.classification_outcome == ClassificationOutcome.TP:
                        unseen_breakdown[cname]["tp"] += 1
                    elif res.classification_outcome == ClassificationOutcome.FN:
                        unseen_breakdown[cname]["fn"] += 1

        metrics = calculate_classification_metrics(tp, fp, fn, tn, unavailable=unavailable)
        per_class_rec = calculate_per_class_recall(class_breakdown)
        unseen_rec = calculate_per_class_recall(unseen_breakdown)

        # Separate scores for distribution tracking
        normal_scores = [r.anomaly_score for r in sample_results if r.is_attack_ground_truth is False]
        attack_scores = [r.anomaly_score for r in sample_results if r.is_attack_ground_truth is True]
        latencies = [r.latency_ms for r in sample_results]

        return {
            "metrics": metrics,
            "per_class_recall": per_class_rec,
            "unseen_class_recall": unseen_rec,
            "sample_results": sample_results,
            "distribution": {
                "normal_scores": normal_scores,
                "attack_scores": attack_scores,
            },
            "latency": {
                "mean_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
                "p95_latency_ms": round(float(sorted(latencies)[int(len(latencies) * 0.95)]), 3) if latencies else 0.0,
            },
        }

