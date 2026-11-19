"""Leave-One-Class-Out (LOCO) Research Experiment Execution Engine (Sprint 4)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any
import pandas as pd

from src.application.detection_service import DetectionService
from src.evaluation.dataset_splitter import DatasetSplitter
from src.evaluation.evaluation_report import EvaluationReportGenerator
from src.evaluation.experiment_config import ExperimentConfig, STANDARD_EXPERIMENTS
from src.evaluation.models import DetectorOutput, EvaluationRecord, InferencePayload
from src.evaluation.prediction_evaluator import PredictionEvaluator
from src.evaluation.provenance import ProvenanceTracker
from src.simulation.data.telemetry_resolver import TelemetryResolver


class LocoExperimentRunner:
    """Executes scientifically valid Leave-One-Class-Out (LOCO) experiments with strict ground-truth isolation."""

    def __init__(self, root: str | Path | None = None, service: DetectionService | None = None) -> None:
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
        self.service = service or DetectionService(self.root, self.root / "data" / "resinet-events.sqlite3")
        self.resolver = TelemetryResolver(self.root)
        self.splitter = DatasetSplitter()

    def run_experiment(
        self,
        config: ExperimentConfig,
        max_test_samples: int | None = None,
    ) -> dict[str, Any]:
        """Execute a full LOCO experiment following strict Phase 4 scientific isolation."""
        domain = config.domain.upper()
        held_out = config.held_out_attack

        # 1. Load dataset & prepare leakage-safe split
        if "IT" in domain:
            data_file = self.root / "data" / "processed" / config.dataset
            df = pd.read_csv(data_file)
            manifest = self.splitter.split_loco_it(
                df,
                held_out_class=held_out or "Botnet ARES",
                random_seed=config.random_seed,
                dataset_name=config.dataset,
                experiment_id=config.experiment_id,
            )
            detector = self.service.detector("IT")
            schema = detector.feature_schema("all_features") if hasattr(detector, "feature_schema") else self.resolver.it_feature_schema()
            threshold = getattr(detector, "_ae_threshold", 0.638311)
            model_name = "IDS2025 Random Forest / Autoencoder"
        else:
            data_file = self.root / "data" / "processed" / config.dataset
            df = pd.read_csv(data_file)
            manifest = self.splitter.split_ot_temporal(
                df,
                random_seed=config.random_seed,
                dataset_name=config.dataset,
                experiment_id=config.experiment_id,
            )
            detector = self.service.detector("OT")
            schema = getattr(detector, "feature_schema", self.resolver.ot_feature_schema())
            threshold = getattr(detector, "threshold", 0.821954)
            model_name = "HAI Autoencoder"

        # Save manifest
        manifest_dir = self.root / "data" / "evaluation" / "manifests"
        manifest.save(manifest_dir)

        # 2. Select test partition indices
        test_indices = manifest.test_indices
        if max_test_samples is not None and max_test_samples > 0:
            test_indices = test_indices[:max_test_samples]

        predictions: list[DetectorOutput] = []
        ground_truths: list[EvaluationRecord] = []

        # 3. Execution loop: STRICT GROUND-TRUTH ISOLATION (Phase 4.3)
        now_dt = datetime.now(timezone.utc)
        for idx in test_indices:
            row = df.iloc[idx]
            event_id = f"{config.experiment_id}-TEST-{idx}"

            # Step A: Construct feature dict matching detector's exact schema
            feat_dict = {}
            for col in schema:
                val = float(row.get(col, 0.0))
                # Reject/sanitize Inf or NaN values
                if pd.isna(val) or pd.isna(val) or abs(val) > 1e12:
                    val = 0.0
                feat_dict[col] = val

            # Step B: Construct InferencePayload (STRICTLY NO LABELS)
            payload = InferencePayload(
                event_id=event_id,
                source="IT" if "IT" in domain else "OT",
                features=feat_dict,
                telemetry_record_id=idx,
            )

            # Step C: Invoke detector
            t_start = time.perf_counter()
            detect_kwargs = {
                "timestamp": now_dt,
                "asset_id": "ASSET_TEST",
                "device_id": "DEV_TEST",
            }
            raw_res = detector.detect(payload.features, **detect_kwargs)
            t_latency = (time.perf_counter() - t_start) * 1000.0

            # Wrap in standardized DetectorOutput
            is_anom = bool(raw_res.anomaly_decision)
            if raw_res.prediction.upper() not in ("NORMAL", "BENIGN"):
                is_anom = True

            out = DetectorOutput(
                source=payload.source,
                detector_name=raw_res.detector,
                prediction=raw_res.prediction,
                is_anomaly=is_anom,
                anomaly_score=float(raw_res.anomaly_score),
                threshold=float(raw_res.threshold) if raw_res.threshold else float(threshold),
                confidence=float(raw_res.confidence) if raw_res.confidence is not None else None,
                feature_schema_version=f"{len(schema)}-features",
                model_version="v2.1",
                latency_ms=t_latency,
            )
            predictions.append(out)

            # Step D: Retrieve evaluation record strictly post-inference
            lbl = str(row.get("label", row.get("Attack", "Normal"))).strip()
            is_atk = lbl.lower() not in ("normal", "benign", "0", "false")
            is_unseen = (lbl.lower() == str(held_out).lower()) if held_out else False

            eval_rec = EvaluationRecord(
                event_id=event_id,
                source=payload.source,
                ground_truth_label=lbl,
                is_attack_ground_truth=is_atk,
                attack_family=lbl,
                is_held_out_unseen=is_unseen,
            )
            ground_truths.append(eval_rec)

        # 4. Evaluate batch results
        evaluation_results = PredictionEvaluator.evaluate_batch(predictions, ground_truths)

        # 5. Collect provenance metadata
        provenance = ProvenanceTracker.get_experiment_provenance(
            root=self.root,
            dataset_name=config.dataset,
            model_name=model_name,
            feature_schema_version=f"{len(schema)}-features",
            threshold=threshold,
            held_out_class=held_out,
            random_seed=config.random_seed,
        )
        provenance["manifest_id"] = manifest.manifest_id
        provenance["manifest_hash"] = manifest.manifest_hash

        # 6. Save reports
        reports_dir = self.root / "data" / "evaluation" / "reports"
        json_path, csv_path = EvaluationReportGenerator.save_reports(
            output_dir=reports_dir,
            experiment_id=config.experiment_id,
            provenance=provenance,
            evaluation_results=evaluation_results,
        )

        return {
            "experiment_id": config.experiment_id,
            "manifest": manifest,
            "provenance": provenance,
            "results": evaluation_results,
            "reports": {
                "json_path": str(json_path),
                "csv_path": str(csv_path),
            },
        }

