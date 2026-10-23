
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DATA_PATH = "data/processed/hai-train-scaled.csv"
MODEL_DIR = "models"


def main():
    print("Loading OT training data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["timestamp"], errors="ignore")

    if X.empty or X.isnull().any().any():
        raise ValueError("Training data is empty or contains missing values.")

    # Preserve time order: earlier records for training,
    # later records for validation.
    split_index = int(len(X) * 0.8)
    X_fit = X.iloc[:split_index]
    X_val = X.iloc[split_index:]

    print(f"Model-training rows: {len(X_fit):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Features: {X.shape[1]}")

    print("\nTraining a separate validation model...")
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=2
    )
    model.fit(X_fit)

    print("Calculating training scores...")
    fit_scores = model.decision_function(X_fit)

    print("Calculating validation scores...")
    val_scores = model.decision_function(X_val)

    # Lower scores indicate more anomalous records.
    thresholds = {
        "1_percent": float(np.quantile(fit_scores, 0.01)),
        "2_percent": float(np.quantile(fit_scores, 0.02)),
        "5_percent": float(np.quantile(fit_scores, 0.05))
    }

    print("\nThreshold analysis (validation data has no labels):")
    for name, threshold in thresholds.items():
        flagged = int(np.sum(val_scores < threshold))
        rate = flagged / len(val_scores) * 100

        print(
            f"{name}: threshold={threshold:.6f}, "
            f"flagged={flagged:,}/{len(val_scores):,} "
            f"({rate:.2f}%)"
        )

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(
        MODEL_DIR, "isolation_forest_validation.joblib"
    )
    threshold_path = os.path.join(
        MODEL_DIR, "validation_thresholds.json"
    )

    joblib.dump(model, model_path)

    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=4)

    print(f"\nValidation model saved: {model_path}")
    print(f"Thresholds saved: {threshold_path}")


if __name__ == "__main__":
    main()
