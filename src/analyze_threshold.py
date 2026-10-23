
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

TRAIN_PATH = "data/processed/hai-train-scaled.csv"
TEST_PATH = "data/processed/hai-test-scaled.csv"
MODEL_PATH = "models/isolation_forest.joblib"


def main():
    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH)
    X_train = train_df.drop(columns=["timestamp"], errors="ignore")

    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)
    y_test = test_df["label"].astype(int)
    X_test = test_df.drop(columns=["timestamp", "label"], errors="ignore")

    # Higher decision_function scores indicate more normal observations.
    train_scores = model.decision_function(X_train)
    test_scores = model.decision_function(X_test)

    # Explore thresholds derived ONLY from training scores.
    # Lower thresholds flag fewer observations as anomalous.
    thresholds = [
        train_scores.min(),
        *pd.Series(train_scores).quantile(
            [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
        ).tolist()
    ]

    print("\nThreshold comparison on the labeled test set")
    print("Threshold\tPrecision\tRecall\t\tF1")

    for threshold in sorted(set(thresholds)):
        # Scores below threshold are classified as anomalies.
        y_pred = (test_scores < threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(
            f"{threshold:.6f}\t"
            f"{precision:.4f}\t\t"
            f"{recall:.4f}\t\t"
            f"{f1:.4f}"
        )


if __name__ == "__main__":
    main()
