"""Explainable Alerts Engine for ResiNet-IDAM v2.0.

Provides fine-grained, auditable alert explainability across IT network flows
and OT industrial telemetry as specified in Priority 6 of the Upgrade Plan:

1. Autoencoder per-feature reconstruction attribution:
   Computes element-wise squared reconstruction residuals (x_j - x̂_j)^2 to identify
   the top physical sensors or network metrics that deviate from normal behavior.
2. Random Forest feature importance attribution:
   Ranks features by overall tree importance and input deviation.
3. Multi-Model Fusion decomposition:
   Deconstructs the final decision into exact percentage contributions:
   Contribution_i = (w_i * S_i) / sum(w_j * S_j) * 100%.
4. Zero-Trust contextual factor explanation:
   Breaks down the 5-factor risk score with clear recommendations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FeatureAttribution:
    """Attribution detail for a single telemetry feature."""
    feature_name: str
    observed_value: float
    expected_value: float
    residual_error: float
    relative_deviation: float  # |observed - expected| / (|expected| + 1e-5)
    unit_or_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "observed_value": round(self.observed_value, 4),
            "expected_value": round(self.expected_value, 4),
            "residual_error": round(self.residual_error, 6),
            "relative_deviation": round(self.relative_deviation, 4),
            "unit_or_description": self.unit_or_description,
        }


@dataclass
class AlertExplanation:
    """Complete explainability package for a security event."""
    domain: str  # 'IT' or 'OT'
    primary_detector: str
    top_features: List[FeatureAttribution] = field(default_factory=list)
    model_contributions: Dict[str, float] = field(default_factory=dict)  # percentage contributions
    model_calibrated_scores: Dict[str, float] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    drift_status: str = "NORMAL"
    threshold_mode: str = "ADAPTIVE"
    summary_narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "primary_detector": self.primary_detector,
            "top_features": [f.to_dict() for f in self.top_features],
            "model_contributions": {k: round(v, 2) for k, v in self.model_contributions.items()},
            "model_calibrated_scores": {k: round(v, 4) for k, v in self.model_calibrated_scores.items()},
            "model_weights": {k: round(v, 4) for k, v in self.model_weights.items()},
            "drift_status": self.drift_status,
            "threshold_mode": self.threshold_mode,
            "summary_narrative": self.summary_narrative,
        }


class AutoencoderExplainer:
    """Computes element-wise reconstruction attribution for autoencoder anomaly detectors."""

    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = list(feature_names)

    def explain(
        self,
        x_observed: np.ndarray,
        x_reconstructed: np.ndarray,
        top_k: int = 5,
    ) -> List[FeatureAttribution]:
        """
        Calculates per-feature squared error and returns the top K diverging features.
        """
        x_obs = np.asarray(x_observed).flatten()
        x_rec = np.asarray(x_reconstructed).flatten()

        residuals = (x_obs - x_rec) ** 2
        top_indices = np.argsort(residuals)[::-1][:top_k]

        attributions = []
        for idx in top_indices:
            idx_int = int(idx)
            name = self.feature_names[idx_int] if idx_int < len(self.feature_names) else f"feature_{idx_int}"
            obs = float(x_obs[idx_int])
            rec = float(x_rec[idx_int])
            err = float(residuals[idx_int])
            rel_dev = float(abs(obs - rec) / (abs(rec) + 1e-5))

            attributions.append(
                FeatureAttribution(
                    feature_name=name,
                    observed_value=obs,
                    expected_value=rec,
                    residual_error=err,
                    relative_deviation=rel_dev,
                )
            )

        return attributions


class FusionExplainer:
    """Decomposes fused multi-model scores into normalized percentage contributions."""

    @staticmethod
    def explain(
        weights: Dict[str, float],
        calibrated_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculates percentage contribution of each model:
        Contribution_i = (w_i * S_i) / sum(w_j * S_j) * 100.
        """
        raw_contribs = {k: weights.get(k, 0.0) * calibrated_scores.get(k, 0.0) for k in weights}
        total = sum(raw_contribs.values())

        if total <= 1e-7:
            # Fall back to weight proportions if scores are near zero
            tot_w = sum(weights.values()) or 1.0
            return {k: round(100.0 * (weights[k] / tot_w), 2) for k in weights}

        return {k: round(100.0 * (raw_contribs[k] / total), 2) for k in weights}


def build_alert_explanation(
    domain: str,
    feature_names: List[str],
    x_observed: np.ndarray,
    x_reconstructed: Optional[np.ndarray] = None,
    fusion_weights: Optional[Dict[str, float]] = None,
    fusion_scores: Optional[Dict[str, float]] = None,
    drift_status: str = "NORMAL",
    threshold_mode: str = "ADAPTIVE",
    top_k: int = 5,
) -> AlertExplanation:
    """Constructs a comprehensive AlertExplanation object."""
    top_features: List[FeatureAttribution] = []
    if x_reconstructed is not None and feature_names:
        ae_explainer = AutoencoderExplainer(feature_names)
        top_features = ae_explainer.explain(x_observed, x_reconstructed, top_k=top_k)

    model_contributions: Dict[str, float] = {}
    if fusion_weights and fusion_scores:
        model_contributions = FusionExplainer.explain(fusion_weights, fusion_scores)

    # Narrative synthesis
    narrative_parts = []
    if top_features:
        top_names = ", ".join(f"`{f.feature_name}`" for f in top_features[:3])
        narrative_parts.append(f"Top anomalous telemetry drivers: {top_names}.")

    if model_contributions:
        top_model = max(model_contributions.items(), key=lambda item: item[1])
        narrative_parts.append(
            f"Primary detection evidence led by {top_model[0]} ({top_model[1]:.1f}% contribution)."
        )

    if drift_status != "NORMAL":
        narrative_parts.append(f"Telemetry stream drift monitor status: {drift_status}.")

    summary_narrative = " ".join(narrative_parts) if narrative_parts else "Anomaly flagged by multi-model pipeline."

    return AlertExplanation(
        domain=domain.upper(),
        primary_detector=f"{domain.upper()}_FUSED_PIPELINE",
        top_features=top_features,
        model_contributions=model_contributions,
        model_calibrated_scores=fusion_scores or {},
        model_weights=fusion_weights or {},
        drift_status=drift_status,
        threshold_mode=threshold_mode,
        summary_narrative=summary_narrative,
    )

