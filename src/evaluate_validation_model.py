
import json
import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

TEST_PATH = "data/processed/hai-test-scaled.csv"
MODEL_PATH = "models/isolation_forest_validation.joblib"
THRESHOLD_PATH = "models/validation_thresholds.json"


def main():
    print("Loading validation model and thresholds...")

    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        thresholds = json.load(file)

    print("Loading labeled OT test data...")
    df = pd.read_csv(TEST_PATH)

    y_true = df["label"].astype(int)
    X_test = df.drop(
        columns=["timestamp", "label"],
        errors="ignore"
    )

    scores = model.decision_function(X_test)

    print(f"Test records: {len(X_test):,}")
    print(f"Features: {X_test.shape[1]}")

    print("\nEvaluation by threshold")
    print("-" * 85)

    for name, threshold in thresholds.items():
        # Isolation Forest scores below the threshold are anomalies.
        y_pred = (scores < threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_true, y_pred, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, zero_division=0
        )
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        print(f"\nThreshold rule: {name}")
        print(f"Threshold:       {threshold:.6f}")
        print(f"True positives:  {tp:,}")
        print(f"False positives: {fp:,}")
        print(f"True negatives:  {tn:,}")
        print(f"False negatives: {fn:,}")
        print(f"Precision:       {precision:.4f}")
        print(f"Recall:          {recall:.4f}")
        print(f"F1-score:        {f1:.4f}")
        print(f"False-positive rate: {fpr:.4f}")

    print("\nEvaluation complete. No model files were overwritten.")


if __name__ == "__main__":
    main()
