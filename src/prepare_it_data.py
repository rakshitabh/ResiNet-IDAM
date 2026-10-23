
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "raw" / "IDS2025.xlsx"
OUTPUT_DIR = ROOT / "data" / "processed" / "it"
MODEL_DIR = ROOT / "models" / "it"

TARGET = "newLabel"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading IDS2025 workbook...")
    df = pd.read_excel(INPUT_PATH, sheet_name=0)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing labels. Inspect before proceeding.")

    # Remove exact duplicate rows only.
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"Exact duplicate rows removed: {before - len(df)}")

    # Separate features and labels.
    y_text = df[TARGET].astype(str)
    X = df.drop(columns=[TARGET]).copy()

    # Convert unexpected non-numeric features to numeric where possible.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Convert all infinities to missing values.
    X = X.replace([np.inf, -np.inf], np.nan)

    # The two columns are flow-rate features; negative rates are invalid.
    for col in ["Flow Bytess", "Flow Packetss"]:
        if col in X.columns:
            negative_count = int((X[col] < 0).sum())
            print(f"Negative values marked missing in {col}: {negative_count}")
            X.loc[X[col] < 0, col] = np.nan

    # Preserve all rows and labels; imputation will be fitted on training data.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Imputation and scaling are fitted on training data only.
    preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    X_train_scaled = preprocessing.fit_transform(X_train)
    X_test_scaled = preprocessing.transform(X_test)

    feature_names = X.columns.tolist()
    class_names = label_encoder.classes_.tolist()

    # Save compact NumPy arrays instead of large CSVs.
    np.savez_compressed(
        OUTPUT_DIR / "it_train.npz",
        X=X_train_scaled.astype(np.float32),
        y=y_train.astype(np.int64),
    )
    np.savez_compressed(
        OUTPUT_DIR / "it_test.npz",
        X=X_test_scaled.astype(np.float32),
        y=y_test.astype(np.int64),
    )

    joblib.dump(preprocessing, MODEL_DIR / "preprocessing.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")

    metadata = {
        "source_file": INPUT_PATH.name,
        "original_rows": int(before),
        "rows_after_exact_deduplication": int(len(df)),
        "feature_count": int(len(feature_names)),
        "target_column": TARGET,
        "class_names": class_names,
        "class_to_integer": {
            name: int(index)
            for index, name in enumerate(class_names)
        },
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "preprocessing": [
            "Exact duplicate rows removed",
            "Infinite values converted to missing",
            "Negative values in flow-rate features converted to missing",
            "Median imputation fitted on training data",
            "StandardScaler fitted on training data",
            "Stratified train/test split",
        ],
        "feature_names": feature_names,
    }

    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nPreparation completed.")
    print("Training shape:", X_train_scaled.shape)
    print("Testing shape:", X_test_scaled.shape)
    print("\nTraining class counts:")
    print(pd.Series(y_train).map(
        dict(enumerate(class_names))
    ).value_counts().to_string())
    print("\nTesting class counts:")
    print(pd.Series(y_test).map(
        dict(enumerate(class_names))
    ).value_counts().to_string())
    print("\nOutputs saved under:", OUTPUT_DIR)
    print("Preprocessor and label encoder saved under:", MODEL_DIR)


if __name__ == "__main__":
    main()
