"""IDS2025 IT detector adapter for deployment artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.domain.events import DetectionResult
from src.infrastructure.model_loader import load_json


class Ids2025Detector:
    source = "IT"
    name = "IDS2025 Random Forest"

    def __init__(self, root: Path) -> None:
        self.data_path = root / "data" / "processed" / "it_realistic"
        self.models_path = root / "models" / "it_realistic"
        metadata_path = self.data_path / "metadata.json"
        self.metadata = load_json(metadata_path) if metadata_path.exists() else {}
        self._model: Any = None
        self._preprocessing: Any = None

    def available(self) -> bool:
        return (
            (self.models_path / "random_forest_all.joblib").exists()
            and bool(self.metadata)
            and len(self.feature_schema()) == 79
        )

    def feature_schema(self) -> list[str]:
        if not self.metadata:
            raise RuntimeError("IDS2025 metadata artifact is unavailable")
        return list(self.metadata["configurations"]["all_features"]["features"])

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = joblib.load(self.models_path / "random_forest_all.joblib")
        return self._model

    def _get_preprocessing(self) -> Any:
        if self._preprocessing is None:
            self._preprocessing = joblib.load(
                self.models_path / "preprocessing_all.joblib"
            )
        return self._preprocessing

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
            raise ValueError("IDS2025 features must match the configured feature contract")
        vector = pd.DataFrame(
            [[values[name] for name in schema]], columns=schema
        )
        vector = self._get_preprocessing().transform(vector)
        model = self._get_model()
        prediction = str(model.predict(vector)[0])
        probabilities = model.predict_proba(vector)[0] if hasattr(model, "predict_proba") else None
        confidence = float(np.max(probabilities)) if probabilities is not None else None
        anomaly = prediction.lower() not in {"normal", "benign"}
        return DetectionResult(
            source="IT",
            detector=self.name,
            prediction=prediction,
            anomaly_score=1.0 - confidence if confidence is not None else float(anomaly),
            threshold=0.5,
            anomaly_decision=anomaly,
            confidence=confidence,
            timestamp=timestamp,
            asset_id=asset_id,
            device_id=device_id,
            features=values,
            metadata={"model_version": self.metadata.get("model_version")},
        )
