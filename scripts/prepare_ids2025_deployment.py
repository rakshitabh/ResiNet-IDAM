"""Prepare IDS2025.csv and train the deployment IT detector bundle."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "IDS2025.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "it_realistic"
MODEL_PATH = ROOT / "models" / "it_realistic"
REPORT_PATH = ROOT / "reports" / "it"
LABEL = "newLabel"
RANDOM_STATE = 42


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"IDS2025 dataset not found: {INPUT_PATH}")

    frame = pd.read_csv(INPUT_PATH)
    if LABEL not in frame.columns:
        raise ValueError(f"Required label column is missing: {LABEL}")

    original_rows = len(frame)
    frame[LABEL] = frame[LABEL].astype("string").str.strip()
    frame = frame.dropna(subset=[LABEL])
    frame = frame[frame[LABEL] != ""].drop_duplicates().reset_index(drop=True)
    duplicate_rows = original_rows - len(frame)

    feature_columns = [column for column in frame.columns if column != LABEL]
    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    labels = frame[LABEL].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    preprocessing = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x_train_scaled = preprocessing.fit_transform(x_train)
    x_test_scaled = preprocessing.transform(x_test)

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessing, MODEL_PATH / "preprocessing_all.joblib")
    joblib.dump(model, MODEL_PATH / "random_forest_all.joblib")
    pd.DataFrame(x_train_scaled, columns=feature_columns).assign(newLabel=y_train.to_numpy()).to_csv(
        OUTPUT_PATH / "it_train_all.csv", index=False
    )
    pd.DataFrame(x_test_scaled, columns=feature_columns).assign(newLabel=y_test.to_numpy()).to_csv(
        OUTPUT_PATH / "it_test_all.csv", index=False
    )

    metadata = {
        "dataset_name": "IDS2025",
        "source_file": str(INPUT_PATH.relative_to(ROOT)),
        "label_column": LABEL,
        "feature_count": len(feature_columns),
        "class_names": sorted(labels.unique().tolist()),
        "configurations": {
            "all_features": {
                "features": feature_columns,
                "feature_names": feature_columns,
                "feature_count": len(feature_columns),
            }
        },
        "preprocessing": {
            "duplicate_rows_removed": duplicate_rows,
            "missing_values_imputed": True,
            "imputation": "median fitted on training partition",
            "scaling": "StandardScaler fitted on training partition",
            "split": "stratified 80/20 split",
            "random_state": RANDOM_STATE,
        },
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        },
    }
    (OUTPUT_PATH / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REPORT_PATH / "ids2025_deployment_report.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
