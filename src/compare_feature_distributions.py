
import pandas as pd
import numpy as np

TRAIN_PATH = "data/processed/hai-train-scaled.csv"
TEST_PATH = "data/processed/hai-test-scaled.csv"
OUTPUT_PATH = "data/processed/feature_distribution_report.csv"


def main():
    print("Loading datasets...")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Keep only features shared by both datasets.
    excluded = {"timestamp", "label"}
    train_features = [c for c in train.columns if c not in excluded]
    test_features = [c for c in test.columns if c not in excluded]
    common = [c for c in train_features if c in test_features]

    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Common features: {len(common)}")

    records = []

    for col in common:
        a = pd.to_numeric(train[col], errors="coerce").dropna()
        b = pd.to_numeric(test[col], errors="coerce").dropna()

        train_std = a.std()
        test_std = b.std()

        records.append({
            "feature": col,
            "train_mean": a.mean(),
            "test_mean": b.mean(),
            "mean_difference": b.mean() - a.mean(),
            "train_std": train_std,
            "test_std": test_std,
            "train_min": a.min(),
            "train_max": a.max(),
            "test_min": b.min(),
            "test_max": b.max(),
            "train_missing": int(train[col].isna().sum()),
            "test_missing": int(test[col].isna().sum()),
        })

    report = pd.DataFrame(records)
    report["absolute_mean_difference"] = (
        report["mean_difference"].abs()
    )

    report = report.sort_values(
        "absolute_mean_difference", ascending=False
    )

    report.to_csv(OUTPUT_PATH, index=False)

    print("\nFeatures with the largest absolute mean differences:")
    print(
        report[
            ["feature", "train_mean", "test_mean",
             "mean_difference", "train_std", "test_std"]
        ].head(15).to_string(index=False)
    )

    print(f"\nFull report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
