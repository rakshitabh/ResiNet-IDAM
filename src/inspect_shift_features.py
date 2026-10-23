
import pandas as pd

TRAIN_PATH = "data/processed/hai-train-features.csv"
TEST_PATH = "data/processed/hai-test-labeled.csv"

FEATURES = [
    "P1_PP04SP",
    "P2_VIBTR04",
    "x1002_08_SETPOINT_OUT",
    "P1_PCV02Z",
    "P1_FT03Z",
    "P2_SCST",
]


def main():
    print("Loading original-scale processed datasets...")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    for feature in FEATURES:
        if feature not in train.columns or feature not in test.columns:
            print(f"\n{feature}: missing from one dataset")
            continue

        a = pd.to_numeric(train[feature], errors="coerce")
        b = pd.to_numeric(test[feature], errors="coerce")

        print(f"\n{'=' * 55}")
        print(f"Feature: {feature}")

        for name, values in [
            ("Training", a),
            ("Test", b),
        ]:
            print(f"\n{name}:")
            print(f"  Valid values: {values.notna().sum():,}")
            print(f"  Missing:      {values.isna().sum():,}")
            print(f"  Unique values: {values.nunique(dropna=True):,}")
            print(f"  Mean:         {values.mean():.6f}")
            print(f"  Std:          {values.std():.6f}")
            print(f"  Min:          {values.min():.6f}")
            print(f"  Max:          {values.max():.6f}")

    print("\nInspection complete. No datasets were modified.")


if __name__ == "__main__":
    main()
