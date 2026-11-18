"""IDS2025 IT Network Flow Intrusion Detector for ResiNet-IDAM v2.0."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd

from src.domain.events import DetectionResult
from src.infrastructure.model_loader import ArtifactError, load_json


class NumpyItAutoencoder:
    """Portable NumPy forward-pass inference engine for the 79-feature IT Autoencoder."""

    def __init__(self, weights_path: str | Path) -> None:
        data = np.load(weights_path)
        self.w0, self.b0 = data["w0"], data["b0"]
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.w3, self.b3 = data["w3"], data["b3"]

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        h0 = np.maximum(0, arr @ self.w0 + self.b0)
        h1 = np.maximum(0, h0 @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        recon = h2 @ self.w3 + self.b3
        return recon


class Ids2025Detector:
    source = "IT"
    name = "IDS2025 Random Forest"
    PORT_COLUMNS = ("Source Port", "Destination Port")

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
        self.metadata_path = self.root / "data" / "processed" / "it_realistic" / "metadata.json"
        self.models_path = self.root / "models" / "it_realistic"
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

        model_path = self._model_path(config_key)
        preprocessor_path = self._preprocessor_path(config_key)
        schema = self.feature_schema(config_key)
        return model_path.exists() and preprocessor_path.exists() and bool(schema)

    def feature_schema(self, config_name: str | None = None) -> list[str]:
        config_key = config_name or self.default_configuration()
        configs = self.configurations()
        if config_key not in configs:
            raise KeyError(f"Unknown configuration '{config_key}'")
        return list(configs[config_key])

    def _model_path(self, config_name: str) -> Path:
        model_name = "all" if config_name == "all_features" else config_name
        return self.models_path / f"random_forest_{model_name}.joblib"

    def _preprocessor_path(self, config_name: str) -> Path:
        model_name = "all" if config_name == "all_features" else config_name
        return self.models_path / f"preprocessing_{model_name}.joblib"

    def _get_model(self, config_name: str) -> Any:
        if config_name not in self._model_cache:
            model_path = self._model_path(config_name)
            if not model_path.exists():
                raise ArtifactError(f"Model artifact not found: {model_path}")
            self._model_cache[config_name] = joblib.load(model_path)
        return self._model_cache[config_name]

    def _get_preprocessing(self, config_name: str) -> Any:
        if config_name not in self._preprocessing_cache:
            preprocessor_path = self._preprocessor_path(config_name)
            if not preprocessor_path.exists():
                raise ArtifactError(f"Preprocessor artifact not found: {preprocessor_path}")
            self._preprocessing_cache[config_name] = joblib.load(preprocessor_path)
        return self._preprocessing_cache[config_name]

    def score(
        self,
        values: dict[str, float],
        config_name: str | None = None,
    ) -> dict[str, Any]:
        config_key = config_name or self.default_configuration()
        schema = self.feature_schema(config_key)
        if list(values) != schema:
            raise ValueError(f"Features must match configuration '{config_key}' in order")

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
        anomaly_decision = prediction.lower() not in {"normal", "benign"}

        return {
            "prediction": prediction,
            "anomaly_score": float(anomaly_decision),
            "threshold": 0.5,
            "anomaly_decision": anomaly_decision,
            "confidence": confidence,
            "reconstruction_error": 1.0 - confidence if confidence is not None else 0.0,
        }

    def top_features(
        self,
        config_name: str | None = None,
        top_n: int = 5,
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
        top_features = []
        ae_decision = False
        ae_score = 0.0
        if self._ae is not None and vector.shape[1] == 79:
            recon = self._ae.predict(vector)
            reconstruction_error = float(np.mean((vector - recon) ** 2))
            threshold = self._ae_threshold
            ae_decision = bool(reconstruction_error > threshold)
            ae_score = min(1.0, max(0.0, reconstruction_error / max(3.0 * threshold, 1e-6)))

            vec_arr = np.asarray(vector).flatten()
            recon_arr = np.asarray(recon).flatten()
            residuals = (vec_arr - recon_arr) ** 2
            top_indices = np.argsort(residuals)[::-1][:5]
            top_features = [
                {
                    "feature_name": schema[idx],
                    "observed_value": round(float(vec_arr[idx]), 4),
                    "expected_value": round(float(recon_arr[idx]), 4),
                    "residual_error": round(float(residuals[idx]), 6),
                    "relative_deviation": round(float(abs(vec_arr[idx] - recon_arr[idx]) / (abs(recon_arr[idx]) + 1e-5)), 4),
                }
                for idx in top_indices
            ]
        else:
            reconstruction_error = 1.0 - confidence if confidence is not None else 0.0
            threshold = 0.5
            if hasattr(model, "feature_importances_"):
                top_imp = self.top_features(config_key=config_key, top_n=5)
                top_features = [
                    {
                        "feature_name": feat,
                        "observed_value": round(float(ordered.get(feat, 0.0)), 4),
                        "expected_value": 0.0,
                        "residual_error": round(float(imp), 6),
                        "relative_deviation": 1.0,
                    }
                    for feat, imp in top_imp
                ]

        # Multi-model detection logic: supervised RF prediction + unsupervised AE
        is_attack_pred = prediction.lower() not in {"normal", "benign"}
        prob_attack = 0.0
        if probabilities is not None:
            normal_idx = 4 if len(probabilities) > 4 else None
            if normal_idx is not None and normal_idx < len(probabilities):
                prob_attack = float(1.0 - probabilities[normal_idx])
            else:
                prob_attack = float(confidence) if is_attack_pred else 0.0

        anomaly_decision = ae_decision or is_attack_pred or (prob_attack >= 0.50)
        if is_attack_pred or prob_attack >= 0.50:
            anomaly_score = max(ae_score, prob_attack)
        elif self._ae is not None:
            anomaly_score = ae_score
        else:
            anomaly_score = float(anomaly_decision)

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
                "top_contributing_features": top_features,
            },
        )
