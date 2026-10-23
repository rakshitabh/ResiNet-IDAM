
import pandas as pd

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
    print("Loading labeled test data...")
    df = pd.read_csv(TEST_PATH)

    print("\nTest label distribution:")
    print(df["label"].value_counts().sort_index().rename(
        index={0: "Normal", 1: "Attack"}
    ))

    for feature in FEATURES:
        if feature not in df.columns:
            print(f"\nSkipping missing feature: {feature}")
            continue

        print(f"\n{'=' * 55}")
        print(f"Feature: {feature}")

        for label, name in [(0, "Normal"), (1, "Attack")]:
            values = pd.to_numeric(
                df.loc[df["label"] == label, feature],
                errors="coerce"
            ).dropna()

            print(f"\n{name}:")
            print(f"  Records: {len(values):,}")
            print(f"  Unique values: {values.nunique():,}")
            print(f"  Mean: {values.mean():.6f}")
            print(f"  Std: {values.std():.6f}")
            print(f"  Min: {values.min():.6f}")
            print(f"  Max: {values.max():.6f}")

    print("\nAnalysis complete. No files were modified.")


if __name__ == "__main__":
    main()
