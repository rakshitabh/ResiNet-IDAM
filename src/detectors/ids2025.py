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


class NumpyItAutoencoder:
    """Portable zero-dependency forward pass for the 4-layer IT Autoencoder."""

    def __init__(self, weights_path: Path) -> None:
        data = np.load(weights_path)
        self.w0 = data["w0"]
        self.b0 = data["b0"]
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.w3 = data["w3"]
        self.b3 = data["b3"]

    def predict(self, vector: np.ndarray) -> np.ndarray:
        x = np.maximum(0, vector @ self.w0 + self.b0)
        x = np.maximum(0, x @ self.w1 + self.b1)
        x = np.maximum(0, x @ self.w2 + self.b2)
        return x @ self.w3 + self.b3


class Ids2025Detector:
    source = "IT"
    name = "IDS2025 Random Forest"
    PORT_COLUMNS = {"Source Port", "Destination Port"}

    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_path = root / "data" / "processed" / "it_realistic"
        self.models_path = root / "models" / "it_realistic"
        self.metadata_path = self.data_path / "metadata.json"
        self.metadata = load_json(self.metadata_path) if self.metadata_path.exists() else {}
        self._model_cache: dict[str, Any] = {}
        self._preprocessing_cache: dict[str, Any] = {}

        self.ae_weights_path = self.models_path / "it_autoencoder_weights.npz"
        self.ae_threshold_path = self.models_path / "it_autoencoder_threshold.json"
        self._ae: NumpyItAutoencoder | None = None
        self._ae_threshold: float = 0.638311
        if self.ae_weights_path.exists():
            self._ae = NumpyItAutoencoder(self.ae_weights_path)
        if self.ae_threshold_path.exists():
            try:
                ae_meta = load_json(self.ae_threshold_path)
                self._ae_threshold = float(ae_meta.get("threshold", 0.638311))
            except Exception:
                pass

    def configurations(self) -> dict[str, list[str]]:
        configs: dict[str, list[str]] = {}

        if self.metadata:
            for name, payload in self.metadata.get("configurations", {}).items():
                features = payload.get("features") or payload.get("feature_names") or []
                if features:
                    configs[str(name)] = list(features)

        if not configs:
            csv_path = self.root / "data" / "processed" / "IDS2025.csv"
            if csv_path.exists():
                frame = pd.read_csv(csv_path)
                if "newLabel" in frame.columns:
                    all_features = [column for column in frame.columns if column != "newLabel"]
                    configs["all_features"] = all_features
                    configs["no_ports"] = [
                        column for column in all_features if column not in self.PORT_COLUMNS
                    ]

        return configs

    def default_configuration(self) -> str:
        configs = self.configurations()
        if "all_features" in configs:
            return "all_features"
        if "no_ports" in configs:
            return "no_ports"
        if configs:
            return next(iter(configs))
        raise RuntimeError("No IDS2025 feature configuration is available")

    def available(self, config_name: str | None = None) -> bool:
        try:
            config_key = config_name or self.default_configuration()
        except RuntimeError:
            return False

        configs = self.configurations()
        if config_key not in configs:
            return False

        model_name = "all" if config_key == "all_features" else config_key
        model_path = self.models_path / f"random_forest_{model_name}.joblib"
        preprocessing_path = self.models_path / f"preprocessing_{model_name}.joblib"
        return (
            model_path.exists()
            and preprocessing_path.exists()
            and len(self.feature_schema(config_key)) > 0
        )

    def feature_schema(self, config_name: str | None = None) -> list[str]:
        config_key = config_name or self.default_configuration()
        configs = self.configurations()
        if config_key not in configs:
            raise RuntimeError(f"IDS2025 configuration is unavailable: {config_key}")
        return list(configs[config_key])

    def _model_path(self, config_name: str | None = None) -> Path:
        config_key = config_name or self.default_configuration()
        model_name = "all" if config_key == "all_features" else config_key
        return self.models_path / f"random_forest_{model_name}.joblib"

    def _preprocessing_path(self, config_name: str | None = None) -> Path:
        config_key = config_name or self.default_configuration()
        model_name = "all" if config_key == "all_features" else config_key
        return self.models_path / f"preprocessing_{model_name}.joblib"

    def _get_model(self, config_name: str | None = None) -> Any:
        config_key = config_name or self.default_configuration()
        if config_key not in self._model_cache:
            self._model_cache[config_key] = joblib.load(self._model_path(config_key))
        return self._model_cache[config_key]

    def _get_preprocessing(self, config_name: str | None = None) -> Any:
        config_key = config_name or self.default_configuration()
        if config_key not in self._preprocessing_cache:
            self._preprocessing_cache[config_key] = joblib.load(
                self._preprocessing_path(config_key)
            )
        return self._preprocessing_cache[config_key]

    def predict_proba(
        self,
        values: dict[str, float],
        *,
        config_name: str | None = None,
    ) -> dict[str, float]:
        config_key = config_name or self.default_configuration()
        schema = self.feature_schema(config_key)
        ordered = {name: float(values[name]) for name in schema}
        vector = pd.DataFrame([[ordered[name] for name in schema]], columns=schema)
        processed = self._get_preprocessing(config_key).transform(vector)
        model = self._get_model(config_key)
        probabilities = model.predict_proba(processed)[0]
        class_names = self.metadata.get("class_names") or []

        def _name_for(idx: Any) -> str:
            try:
                i = int(idx)
                if 0 <= i < len(class_names):
                    return class_names[i]
            except (ValueError, TypeError):
                pass
            return str(idx)

        return {
            _name_for(label): float(prob)
            for label, prob in zip(model.classes_, probabilities)
        }

    def feature_importance(
        self,
        config_name: str | None = None,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        config_key = config_name or self.default_configuration()
        schema = self.feature_schema(config_key)
        model = self._get_model(config_key)
        if not hasattr(model, "feature_importances_"):
            return []
        ranked = sorted(
            zip(schema, model.feature_importances_),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return ranked[:top_n]

    def detect(
        self,
        values: dict[str, float],
        *,
        timestamp: datetime,
        asset_id: str,
        device_id: str,
        config_name: str | None = None,
    ) -> DetectionResult:
        config_key = config_name or self.default_configuration()
        schema = self.feature_schema(config_key)
        ordered = {name: float(values[name]) for name in schema}
        vector = pd.DataFrame([[ordered[name] for name in schema]], columns=schema)
        vector = self._get_preprocessing(config_key).transform(vector)
        model = self._get_model(config_key)

        class_names = self.metadata.get("class_names") or []
        raw_pred = model.predict(vector)[0]
        try:
            pred_idx = int(raw_pred)
            prediction = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(raw_pred)
        except (ValueError, TypeError):
            prediction = str(raw_pred)

        probabilities = model.predict_proba(vector)[0] if hasattr(model, "predict_proba") else None
        confidence = float(np.max(probabilities)) if probabilities is not None else None

        # Compute unsupervised Autoencoder anomaly score if available
        if self._ae is not None and vector.shape[1] == 79:
            recon = self._ae.predict(vector)
            reconstruction_error = float(np.mean((vector - recon) ** 2))
            threshold = self._ae_threshold
            anomaly_decision = reconstruction_error > threshold
            # Normalized score bounded to [0.0, 1.0] for Zero-Trust risk engine:
            anomaly_score = min(1.0, max(0.0, reconstruction_error / max(3.0 * threshold, 1e-6)))
        else:
            reconstruction_error = 1.0 - confidence if confidence is not None else 0.0
            anomaly_decision = prediction.lower() not in {"normal", "benign"}
            anomaly_score = float(anomaly_decision)
            threshold = 0.5

        return DetectionResult(
            source="IT",
            detector=self.name,
            prediction=prediction,
            anomaly_score=anomaly_score,
            threshold=threshold,
            anomaly_decision=anomaly_decision,
            confidence=confidence,
            timestamp=timestamp,
            asset_id=asset_id,
            device_id=device_id,
            features=ordered,
            metadata={
                "model_version": self.metadata.get("model_version"),
                "config_name": config_key,
                "reconstruction_error": reconstruction_error,
                "calibrated_threshold": threshold,
                "calibrated_anomaly_score": min(reconstruction_error / threshold, 1.0) if threshold > 0 else 0.0,
                "supervised_prediction": prediction,
                "supervised_confidence": confidence,
                "anomaly_decision": anomaly_decision,
                "temporal_status": "telemetry_lacks_timestamps_and_ips",
                "probabilities": {str(c): float(p) for c, p in zip(model.classes_, probabilities)} if probabilities is not None else {},
            },
        )