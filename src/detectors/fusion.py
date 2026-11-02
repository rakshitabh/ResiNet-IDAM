"""Calibrated Multi-Model Fusion and Zero-Trust Alignment for ResiNet-IDAM v2.0.

Implements Priority 5 from the ResiNet-IDAM Upgrade Plan:
1. Calibrated score normalization:
   Converts raw model outputs (Random Forest probabilities, Autoencoder reconstruction
   errors, Isolation Forest outlier scores) into standardized [0, 1] anomaly confidence
   scores using centered sigmoid calibration around operational decision thresholds.
2. Domain-specific fusion:
   - ITFusionDetector: Calibrated ensemble of Random Forest (supervised),
     Residual Autoencoder (unsupervised reconstruction), and Isolation Forest (density).
     S_IT = w_RF * S_RF + w_AE * S_AE + w_IF * S_IF, with sum(w) = 1.
   - OTFusionDetector: Calibrated ensemble of OT Autoencoder and OT Isolation Forest.
     S_OT = w_AE * S_AE + w_IF * S_IF, with sum(w) = 1.
3. Adaptive Weighting:
   - Updates weights based on verified feedback error rates.
   - Detects zero-day divergence (high unsupervised anomaly score when supervised RF is low)
     and dynamically boosts unsupervised detector influence.
4. Zero-Trust Alignment:
   Directly transforms fused calibrated outputs into inputs for ZeroTrustRiskEngine,
   ensuring continuous, explainable, risk-tiered policy enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.zero_trust.risk_engine import RiskInput, calculate_risk


class FusionMode(str, Enum):
    STATIC = "STATIC"
    ADAPTIVE = "ADAPTIVE"


@dataclass
class FusedDetectionResult:
    """Detailed output of a multi-model fusion evaluation."""
    domain: str  # 'IT' or 'OT'
    fused_score: float  # [0.0, 1.0]
    is_anomaly: bool
    decision_threshold: float
    weights: Dict[str, float]
    calibrated_scores: Dict[str, float]
    raw_scores: Dict[str, float]
    contributions: Dict[str, float]  # w_i * S_i
    zero_day_suspected: bool = False
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "fused_score": round(self.fused_score, 6),
            "is_anomaly": self.is_anomaly,
            "decision_threshold": round(self.decision_threshold, 6),
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "calibrated_scores": {k: round(v, 4) for k, v in self.calibrated_scores.items()},
            "raw_scores": {k: round(v, 6) for k, v in self.raw_scores.items()},
            "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
            "zero_day_suspected": self.zero_day_suspected,
        }


def sigmoid_calibrate(value: Union[float, np.ndarray], threshold: float, gamma: float = 5.0) -> Union[float, np.ndarray]:
    """
    Transforms an unbounded or uncalibrated anomaly score into [0.0, 1.0].
    
    Calibration properties:
    - At value == threshold: S = 0.50 (the decision boundary)
    - For value >> threshold: S -> 1.00 (high anomaly confidence)
    - For value << threshold: S -> 0.00 (high benign confidence)
    """
    diff = np.clip(gamma * (value - threshold), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-diff))


class ITFusionDetector:
    """
    Calibrated multi-model fusion detector for enterprise IT network flows.
    
    Combines:
    1. Random Forest (Supervised classification -> attack probability)
    2. Residual Autoencoder (Unsupervised manifold reconstruction)
    3. Isolation Forest (Unsupervised density outlier)
    """

    def __init__(
        self,
        weights: Optional[Tuple[float, float, float]] = None,
        ae_threshold: float = 0.638311,
        ae_gamma: float = 5.0,
        if_threshold: float = 0.4500,
        if_gamma: float = 10.0,
        decision_threshold: float = 0.50,
        mode: FusionMode = FusionMode.STATIC,
    ) -> None:
        # Default static weights: RF=0.60, AE=0.20, IF=0.20
        if weights is None:
            weights = (0.60, 0.20, 0.20)
        w_rf, w_ae, w_if = weights
        total = w_rf + w_ae + w_if
        self.weights = {
            "RF": float(w_rf / total),
            "ResidualAE": float(w_ae / total),
            "IF": float(w_if / total),
        }
        self.base_weights = dict(self.weights)

        self.ae_threshold = float(ae_threshold)
        self.ae_gamma = float(ae_gamma)
        self.if_threshold = float(if_threshold)
        self.if_gamma = float(if_gamma)
        self.decision_threshold = float(decision_threshold)
        self.mode = mode

        # Feedback verification statistics
        self._feedback_history: List[Dict[str, Any]] = []

    def calibrate_scores(
        self,
        rf_prob_attack: float,
        ae_mse: float,
        if_raw_score: float,
    ) -> Dict[str, float]:
        """Converts raw model outputs to calibrated [0, 1] anomaly scores."""
        # RF attack probability is already naturally in [0, 1]
        s_rf = float(np.clip(rf_prob_attack, 0.0, 1.0))
        # AE MSE calibrated via sigmoid around ae_threshold
        s_ae = float(sigmoid_calibrate(ae_mse, self.ae_threshold, self.ae_gamma))
        # IF score (-score_samples) calibrated via sigmoid around if_threshold
        s_if = float(sigmoid_calibrate(if_raw_score, self.if_threshold, self.if_gamma))

        return {
            "RF": s_rf,
            "ResidualAE": s_ae,
            "IF": s_if,
        }

    def fuse(
        self,
        rf_prob_attack: float,
        ae_mse: float,
        if_raw_score: float,
        enable_zero_day_boost: bool = True,
    ) -> FusedDetectionResult:
        """
        Calculates the fused anomaly score and per-model contributions.
        """
        calibrated = self.calibrate_scores(rf_prob_attack, ae_mse, if_raw_score)
        raw = {"RF": rf_prob_attack, "ResidualAE": ae_mse, "IF": if_raw_score}

        current_weights = dict(self.weights)
        zero_day_suspected = False

        # Zero-Day Divergence Safeguard:
        # If supervised RF predicts Benign (<= 0.20), but unsupervised detectors observe
        # severe manifold / density anomalies (>= 0.70), flag potential zero-day and
        # adaptively elevate unsupervised detector influence.
        if enable_zero_day_boost and self.mode == FusionMode.ADAPTIVE:
            if calibrated["RF"] < 0.25 and (calibrated["ResidualAE"] > 0.70 or calibrated["IF"] > 0.70):
                zero_day_suspected = True
                # Rebalance weights towards unsupervised components
                current_weights = {
                    "RF": 0.20,
                    "ResidualAE": 0.40,
                    "IF": 0.40,
                }

        # Compute weighted sum
        fused_score = sum(current_weights[m] * calibrated[m] for m in current_weights)
        fused_score = float(np.clip(fused_score, 0.0, 1.0))

        contributions = {m: current_weights[m] * calibrated[m] for m in current_weights}
        is_anomaly = fused_score >= self.decision_threshold

        return FusedDetectionResult(
            domain="IT",
            fused_score=fused_score,
            is_anomaly=is_anomaly,
            decision_threshold=self.decision_threshold,
            weights=current_weights,
            calibrated_scores=calibrated,
            raw_scores=raw,
            contributions=contributions,
            zero_day_suspected=zero_day_suspected,
        )

    def update_weights_from_feedback(self, verified_samples: List[Dict[str, Any]], alpha: float = 0.80) -> None:
        """
        Updates weights based on human-verified ground-truth feedback.
        Each sample dict must contain: 'y_true' (0 or 1), 'rf_pred', 'ae_pred', 'if_pred'.
        """
        if not verified_samples:
            return

        errors = {"RF": 0.0, "ResidualAE": 0.0, "IF": 0.0}
        n = len(verified_samples)

        for s in verified_samples:
            y = s["y_true"]
            errors["RF"] += abs(y - s["rf_pred"])
            errors["ResidualAE"] += abs(y - s["ae_pred"])
            errors["IF"] += abs(y - s["if_pred"])

        for k in errors:
            errors[k] /= n

        # Reliability r_i = 1 / (error + 0.05)
        reliabilities = {k: 1.0 / (errors[k] + 0.05) for k in errors}
        tot_rel = sum(reliabilities.values())
        target_weights = {k: reliabilities[k] / tot_rel for k in errors}

        # EMA update with safety floor of 0.05
        for k in self.weights:
            updated = alpha * self.weights[k] + (1.0 - alpha) * target_weights[k]
            self.weights[k] = max(0.05, updated)

        tot = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= tot

        self._feedback_history.append({
            "samples_evaluated": n,
            "errors": errors,
            "updated_weights": dict(self.weights),
        })


class OTFusionDetector:
    """
    Calibrated multi-model fusion detector for industrial OT physical telemetry.
    
    Combines:
    1. OT Autoencoder (66-feature deep non-linear reconstruction)
    2. OT Isolation Forest (High-dimensional density outlier)
    """

    def __init__(
        self,
        weights: Optional[Tuple[float, float]] = None,
        ae_threshold: float = 0.821954,
        ae_gamma: float = 2.0,
        if_threshold: float = 0.5284,
        if_gamma: float = 10.0,
        decision_threshold: float = 0.50,
        mode: FusionMode = FusionMode.STATIC,
    ) -> None:
        # Default empirical weights: AE=0.45, IF=0.55 (achieves F1 45.64% vs 39.84% AE and 29.92% IF)
        if weights is None:
            weights = (0.45, 0.55)
        w_ae, w_if = weights
        total = w_ae + w_if
        self.weights = {
            "AE": float(w_ae / total),
            "IF": float(w_if / total),
        }
        self.base_weights = dict(self.weights)

        self.ae_threshold = float(ae_threshold)
        self.ae_gamma = float(ae_gamma)
        self.if_threshold = float(if_threshold)
        self.if_gamma = float(if_gamma)
        self.decision_threshold = float(decision_threshold)
        self.mode = mode
        self._feedback_history: List[Dict[str, Any]] = []

    def calibrate_scores(self, ae_mse: float, if_raw_score: float) -> Dict[str, float]:
        """Converts raw OT model outputs to calibrated [0, 1] anomaly scores."""
        s_ae = float(sigmoid_calibrate(ae_mse, self.ae_threshold, self.ae_gamma))
        s_if = float(sigmoid_calibrate(if_raw_score, self.if_threshold, self.if_gamma))
        return {"AE": s_ae, "IF": s_if}

    def fuse(self, ae_mse: float, if_raw_score: float) -> FusedDetectionResult:
        calibrated = self.calibrate_scores(ae_mse, if_raw_score)
        raw = {"AE": ae_mse, "IF": if_raw_score}

        current_weights = dict(self.weights)
        fused_score = sum(current_weights[m] * calibrated[m] for m in current_weights)
        fused_score = float(np.clip(fused_score, 0.0, 1.0))

        contributions = {m: current_weights[m] * calibrated[m] for m in current_weights}
        is_anomaly = fused_score >= self.decision_threshold

        return FusedDetectionResult(
            domain="OT",
            fused_score=fused_score,
            is_anomaly=is_anomaly,
            decision_threshold=self.decision_threshold,
            weights=current_weights,
            calibrated_scores=calibrated,
            raw_scores=raw,
            contributions=contributions,
            zero_day_suspected=False,
        )

    def update_weights_from_feedback(self, verified_samples: List[Dict[str, Any]], alpha: float = 0.80) -> None:
        if not verified_samples:
            return

        errors = {"AE": 0.0, "IF": 0.0}
        n = len(verified_samples)

        for s in verified_samples:
            y = s["y_true"]
            errors["AE"] += abs(y - s["ae_pred"])
            errors["IF"] += abs(y - s["if_pred"])

        for k in errors:
            errors[k] /= n

        reliabilities = {k: 1.0 / (errors[k] + 0.05) for k in errors}
        tot_rel = sum(reliabilities.values())
        target_weights = {k: reliabilities[k] / tot_rel for k in errors}

        for k in self.weights:
            updated = alpha * self.weights[k] + (1.0 - alpha) * target_weights[k]
            self.weights[k] = max(0.10, updated)

        tot = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= tot

        self._feedback_history.append({
            "samples_evaluated": n,
            "errors": errors,
            "updated_weights": dict(self.weights),
        })


def to_zero_trust_risk_input(
    fused_result: FusedDetectionResult,
    identity_trust: float = 0.85,
    device_trust: float = 0.85,
    asset_criticality: float = 0.70,
    identity_id: str = "operator-console",
    device_id: str = "edge-gateway-01",
    asset_id: str = "primary-subsystem",
) -> RiskInput:
    """
    Transforms a FusedDetectionResult directly into an inspectable RiskInput.
    
    Mapping logic:
    - anomaly_severity = fused_score (calibrated [0, 1])
    - behavior_deviation = max(calibrated_scores) (captures worst-case single-detector alarm)
    """
    anomaly_severity = float(np.clip(fused_result.fused_score, 0.0, 1.0))
    max_dev = max(fused_result.calibrated_scores.values()) if fused_result.calibrated_scores else anomaly_severity
    behavior_deviation = float(np.clip(max_dev, 0.0, 1.0))

    return RiskInput(
        identity_trust=float(identity_trust),
        device_trust=float(device_trust),
        asset_criticality=float(asset_criticality),
        anomaly_severity=anomaly_severity,
        behavior_deviation=behavior_deviation,
        identity_id=identity_id,
        device_id=device_id,
        asset_id=asset_id,
    )


def evaluate_fused_risk(
    fused_result: FusedDetectionResult,
    identity_trust: float = 0.85,
    device_trust: float = 0.85,
    asset_criticality: float = 0.70,
) -> Dict[str, Any]:
    """Calculates full Zero-Trust risk score and factor breakdown for a fused detection event."""
    risk_input = to_zero_trust_risk_input(
        fused_result,
        identity_trust=identity_trust,
        device_trust=device_trust,
        asset_criticality=asset_criticality,
    )
    risk_output = calculate_risk(risk_input)
    risk_output["fusion_detail"] = fused_result.to_dict()
    return risk_output

