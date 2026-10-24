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
        self._metadata = load_json(self.threshold_path)

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

    def detect(
        self,
        values: dict[str, float],
        *,
        timestamp: datetime,
        asset_id: str,
        device_id: str,
    ) -> DetectionResult:
        schema = self.feature_schema()
        if list(values) != schema:
            raise ValueError("HAI features must match the ordered 66-feature contract")
        vector = np.asarray([values[name] for name in schema], dtype=np.float32).reshape(1, -1)
        reconstructed = self._get_model().predict(vector, verbose=0)
        score = float(np.mean(np.square(vector - reconstructed)))
        threshold = float(self._metadata["threshold"])
        return DetectionResult(
            source="OT",
            detector=self.name,
            prediction="ATTACK" if score > threshold else "NORMAL",
            anomaly_score=score,
            threshold=threshold,
            anomaly_decision=score > threshold,
            confidence=min(score / threshold, 1.0),
            timestamp=timestamp,
            asset_id=asset_id,
            device_id=device_id,
            features=values,
            metadata={"model_version": self._metadata.get("model_version")},
        )
