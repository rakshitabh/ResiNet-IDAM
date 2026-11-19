"""Normal-only threshold calibrator for unsupervised and novelty anomaly detectors (Sprint 3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np


class ThresholdCalibrator:
    """Computes and freezes operational anomaly decision thresholds using normal validation data."""

    @staticmethod
    def calibrate_percentile(
        normal_scores: list[float] | np.ndarray,
        target_fpr: float = 0.01,
        frozen_tau_override: float | None = None,
    ) -> dict[str, Any]:
        """Compute threshold tau at (100 - target_fpr*100)th percentile on normal calibration samples."""
        scores_arr = np.asarray(normal_scores, dtype=np.float64)
        if len(scores_arr) == 0:
            raise ValueError("Cannot calibrate threshold on empty normal validation partition")

        if frozen_tau_override is not None:
            tau = float(frozen_tau_override)
            percentile = float(np.mean(scores_arr <= tau) * 100.0)
        else:
            percentile = (1.0 - target_fpr) * 100.0
            tau = float(np.percentile(scores_arr, percentile))

        measured_fpr = float(np.mean(scores_arr > tau))

        return {
            "threshold": tau,
            "target_fpr": float(target_fpr),
            "measured_calibration_fpr": round(measured_fpr, 6),
            "percentile_rank": round(percentile, 2),
            "validation_sample_count": len(scores_arr),
            "mean_score": round(float(np.mean(scores_arr)), 6),
            "max_score": round(float(np.max(scores_arr)), 6),
            "is_frozen": True,
        }

    @classmethod
    def save_calibration(cls, calibration_dict: dict[str, Any], output_path: str | Path) -> None:
        """Persist frozen threshold calibration record to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(calibration_dict, f, indent=2)

    @classmethod
    def load_calibration(cls, file_path: str | Path) -> dict[str, Any]:
        """Load frozen threshold record."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

