"""HAI 21.03 Autoencoder Detector for Industrial OT Physical Telemetry."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except (ImportError, Exception):
    keras = None
    HAS_TENSORFLOW = False

from src.domain.events import DetectionResult
from src.infrastructure.model_loader import ArtifactError, load_json


class NumpyAutoencoder:
    """Portable NumPy forward-pass inference engine for the 66-feature OT Autoencoder."""

    def __init__(self, weights_path: str | Path) -> None:
        data = np.load(weights_path)
        self.w0, self.b0 = data["w0"], data["b0"]
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.w3, self.b3 = data["w3"], data["b3"]

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        h0 = np.maximum(0, X @ self.w0 + self.b0)
        h1 = np.maximum(0, h0 @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        recon = h2 @ self.w3 + self.b3
        return recon


class HaiAutoencoderDetector:
    source = "OT"
    name = "HAI Autoencoder"

    def __init__(self, models_path: Path) -> None:
        self.model_path = models_path / "ot_autoencoder.keras" if (models_path / "ot_autoencoder.keras").exists() else models_path / "autoencoder.keras"
        self.weights_path = models_path / "ot_autoencoder_weights.npz"
        self.threshold_path = models_path / "autoencoder_threshold.json" if (models_path / "autoencoder_threshold.json").exists() else models_path / "anomaly_threshold.json"
        self._model: Any = None
        self._metadata: dict[str, Any] = {}

        if self.threshold_path.exists():
            try:
                self._metadata = load_json(self.threshold_path)
            except ArtifactError:
                self._metadata = {}
        else:
            self._metadata = {
                "feature_columns": [],
                "threshold": 0.0,
                "model_version": "untrained",
            }

    def available(self) -> bool:
        has_model = self.weights_path.exists() or (HAS_TENSORFLOW and self.model_path.exists())
        return has_model and bool(self._metadata.get("feature_columns"))

    def feature_schema(self) -> list[str]:
        return list(self._metadata["feature_columns"])

    def _get_model(self) -> Any:
        if self._model is None:
            if self.weights_path.exists():
                self._model = NumpyAutoencoder(self.weights_path)
            elif HAS_TENSORFLOW and self.model_path.exists():
                self._model = keras.models.load_model(self.model_path)
            else:
                raise ArtifactError(f"Model artifact not found: {self.weights_path} or {self.model_path}")
        return self._model

    def score(self, values: dict[str, float]) -> dict[str, Any]:
        schema = self.feature_schema()
        if list(values) != schema:
            raise ValueError("HAI features must match the ordered 66-feature contract")

        vector = np.asarray([values[name] for name in schema], dtype=np.float32).reshape(1, -1)
        reconstructed = self._get_model().predict(vector, verbose=0)
        residuals = np.square(vector - reconstructed).flatten()
        reconstruction_error = float(np.mean(residuals))
        threshold = float(self._metadata["threshold"])
        prediction = "ATTACK" if reconstruction_error > threshold else "NORMAL"
        confidence = min(reconstruction_error / threshold, 1.0) if threshold else 0.0

        top_indices = np.argsort(residuals)[::-1][:5]
        top_features = [
            {
                "feature_name": schema[idx],
                "observed_value": round(float(vector[0, idx]), 4),
                "expected_value": round(float(reconstructed[0, idx]), 4),
                "residual_error": round(float(residuals[idx]), 6),
                "relative_deviation": round(float(abs(vector[0, idx] - reconstructed[0, idx]) / (abs(reconstructed[0, idx]) + 1e-5)), 4),
            }
            for idx in top_indices
        ]

        return {
            "prediction": prediction,
            "anomaly_score": reconstruction_error,
            "threshold": threshold,
            "anomaly_decision": reconstruction_error > threshold,
            "confidence": confidence,
            "reconstruction_error": reconstruction_error,
            "top_contributing_features": top_features,
        }

    def detect(
        self,
        values: dict[str, float],
        *,
        timestamp: datetime,
        asset_id: str,
        device_id: str,
        config_name: str | None = None,
    ) -> DetectionResult:
        summary = self.score(values)
        return DetectionResult(
            source="OT",
            detector=self.name,
            prediction=summary["prediction"],
            anomaly_score=summary["anomaly_score"],
            threshold=summary["threshold"],
            anomaly_decision=summary["anomaly_decision"],
            confidence=summary["confidence"],
            timestamp=timestamp,
            asset_id=asset_id,
            device_id=device_id,
            features=values,
            metadata={
                "model_version": self._metadata.get("model_version"),
                "reconstruction_error": summary["reconstruction_error"],
                "top_contributing_features": summary["top_contributing_features"],
            },
        )
