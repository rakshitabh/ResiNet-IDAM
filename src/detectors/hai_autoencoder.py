"""HAI OT Autoencoder detector."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras

from src.domain.events import DetectionResult
from src.infrastructure.model_loader import ArtifactError, load_json

class HaiAutoencoderDetector:
    source = "OT"
    name = "HAI Autoencoder"

    def __init__(self, models_path: Path) -> None:
        self.model_path = models_path / "autoencoder.keras"
        self.threshold_path = models_path / "anomaly_threshold.json"
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
        return self.model_path.exists() and bool(self._metadata.get("feature_columns"))




    
    def feature_schema(self) -> list[str]:
        return list(self._metadata["feature_columns"])

    def _get_model(self) -> Any:
        if self._model is None:
            if not self.model_path.exists():
                raise ArtifactError(f"Model artifact not found: {self.model_path}")
            self._model = keras.models.load_model(self.model_path)
        return self._model

    def score(self, values: dict[str, float]) -> dict[str, Any]:
        schema = self.feature_schema()
        if list(values) != schema:
            raise ValueError("HAI features must match the ordered 66-feature contract")

        vector = np.asarray([values[name] for name in schema], dtype=np.float32).reshape(1, -1)
        reconstructed = self._get_model().predict(vector, verbose=0)
        reconstruction_error = float(np.mean(np.square(vector - reconstructed)))
        threshold = float(self._metadata["threshold"])
        prediction = "ATTACK" if reconstruction_error > threshold else "NORMAL"
        confidence = min(reconstruction_error / threshold, 1.0) if threshold else 0.0

        return {
            "prediction": prediction,
            "anomaly_score": reconstruction_error,
            "threshold": threshold,
            "anomaly_decision": reconstruction_error > threshold,
            "confidence": confidence,
            "reconstruction_error": reconstruction_error,
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
            },
        )