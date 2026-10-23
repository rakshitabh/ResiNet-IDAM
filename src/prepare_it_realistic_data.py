"""
ResiNet-IDAM: Realistic IT Dataset Preparation Pipeline
-------------------------------------------------------
Splits each contiguous label block chronologically along the row order:
- Earlier 80% rows of each block -> Training partition
- Later 20% rows of each block   -> Testing partition

Generates two feature configurations:
1. 'all': All 79 numeric network features.
2. 'no_ports': 77 numeric features (Source Port and Destination Port removed).

Strict Preprocessing Isolation:
- Median imputation and StandardScaler are fitted SOLELY on training data.
- Testing data is transformed using the training-fitted pipelines.

Artifacts saved under:
- data/processed/it_realistic/
- models/it_realistic/
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "raw" / "IDS2025.xlsx"
OUTPUT_DIR = ROOT / "data" / "processed" / "it_realistic"
MODEL_DIR = ROOT / "models" / "it_realistic"

TARGET = "newLabel"
PORT_COLUMNS = ["Source Port", "Destination Port"]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=== [1] Loading IDS2025 Raw Workbook ===")
    df = pd.read_excel(INPUT_PATH, sheet_name=0)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing labels. Inspect before proceeding.")

    # 1. Deduplication: exact duplicate rows removed (preserving first-occurrence row order)
    before_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True).copy()
    print(f"Total rows read: {before_len}")
    print(f"Exact duplicate rows removed: {before_len - len(df)}")
    print(f"Total unique rows: {len(df)}")

    # 2. Identify Contiguous Label Blocks along physical row order
    df["block_id"] = (df[TARGET] != df[TARGET].shift(1)).cumsum()
    blocks = df.groupby("block_id").agg(
        label=(TARGET, "first"),
        count=(TARGET, "count")
    ).reset_index()
    print(f"\nTotal contiguous label blocks identified: {len(blocks)}")

    train_indices = []
    test_indices = []
    block_records = []

    # 3. Chronological 80/20 Block Split (Earlier 80% -> Train, Later 20% -> Test)
    for _, b_row in blocks.iterrows():
        b_id = b_row["block_id"]
        grp = df[df["block_id"] == b_id]
        n = len(grp)
        label = b_row["label"]

        if n == 1:
            n_train = 1
        else:
            n_train = int(round(n * 0.8))
            n_train = min(n_train, n - 1)
            n_train = max(n_train, 1)

        n_test = n - n_train
        b_train_idx = grp.index[:n_train].tolist()
        b_test_idx = grp.index[n_train:].tolist()

        train_indices.extend(b_train_idx)
        test_indices.extend(b_test_idx)

        block_records.append({
            "block_id": int(b_id),
            "label": label,
            "total_rows": int(n),
            "train_rows": int(n_train),
            "test_rows": int(n_test),
            "start_raw_index": int(grp.index[0]),
            "split_raw_index": int(grp.index[n_train]) if n_train < n else int(grp.index[-1]),
            "end_raw_index": int(grp.index[-1]),
        })

    print(f"\nContiguous Block Split Summary:")
    print(f"  Training rows: {len(train_indices)} ({len(train_indices)/len(df)*100:.2f}%)")
    print(f"  Testing rows:  {len(test_indices)} ({len(test_indices)/len(df)*100:.2f}%)")

    # 4. Feature and Label Preparation
    y_raw = df[TARGET].astype(str)
    X_raw = df.drop(columns=[TARGET, "block_id"]).copy()

    # Numeric conversion
    for col in X_raw.columns:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

    # Replace infinities with NaN
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

    # Sanitize invalid negative flow rates
    for col in ["Flow Bytess", "Flow Packetss"]:
        if col in X_raw.columns:
            neg_count = int((X_raw[col] < 0).sum())
            if neg_count > 0:
                print(f"  Negative flow rate marked missing in '{col}': {neg_count} rows")
                X_raw.loc[X_raw[col] < 0, col] = np.nan

    # Fit label encoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_.tolist()
    class_to_int = {cls: idx for idx, cls in enumerate(class_names)}

    # Slice train and test dataframes
    X_train_df = X_raw.loc[train_indices].copy()
    X_test_df = X_raw.loc[test_indices].copy()
    y_train = y_encoded[train_indices]
    y_test = y_encoded[test_indices]

    # Verify class distribution
    train_counts = {class_names[c]: int((y_train == c).sum()) for c in range(len(class_names))}
    test_counts = {class_names[c]: int((y_test == c).sum()) for c in range(len(class_names))}
    
    print("\nClass Distribution Across Splits:")
    for cls in class_names:
        tr_c = train_counts[cls]
        te_c = test_counts[cls]
        pct = te_c / (tr_c + te_c) * 100.0
        print(f"  {cls:<15}: Train={tr_c:>5}, Test={te_c:>5} (Test % = {pct:.2f}%)")

    # 5. Configuration A: All 79 numeric features
    print("\n=== [2] Preprocessing Configuration A: All Features (79 features) ===")
    features_all = X_train_df.columns.tolist()
    
    pipe_all = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Fit strictly on training data
    X_train_all_scaled = pipe_all.fit_transform(X_train_df[features_all])
    X_test_all_scaled = pipe_all.transform(X_test_df[features_all])

    np.savez_compressed(
        OUTPUT_DIR / "it_train_all.npz",
        X=X_train_all_scaled.astype(np.float32),
        y=y_train.astype(np.int64),
    )
    np.savez_compressed(
        OUTPUT_DIR / "it_test_all.npz",
        X=X_test_all_scaled.astype(np.float32),
        y=y_test.astype(np.int64),
    )
    joblib.dump(pipe_all, MODEL_DIR / "preprocessing_all.joblib")
    print(f"  Saved: it_train_all.npz {X_train_all_scaled.shape}, it_test_all.npz {X_test_all_scaled.shape}")

    # 6. Configuration B: Port features removed (77 features)
    print("\n=== [3] Preprocessing Configuration B: No Ports (77 features) ===")
    features_no_ports = [col for col in features_all if col not in PORT_COLUMNS]
    assert len(features_no_ports) == len(features_all) - 2, "Failed to remove exactly 2 port columns"

    pipe_no_ports = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Fit strictly on training data
    X_train_no_ports_scaled = pipe_no_ports.fit_transform(X_train_df[features_no_ports])
    X_test_no_ports_scaled = pipe_no_ports.transform(X_test_df[features_no_ports])

    np.savez_compressed(
        OUTPUT_DIR / "it_train_no_ports.npz",
        X=X_train_no_ports_scaled.astype(np.float32),
        y=y_train.astype(np.int64),
    )
    np.savez_compressed(
        OUTPUT_DIR / "it_test_no_ports.npz",
        X=X_test_no_ports_scaled.astype(np.float32),
        y=y_test.astype(np.int64),
    )
    joblib.dump(pipe_no_ports, MODEL_DIR / "preprocessing_no_ports.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")
    print(f"  Saved: it_train_no_ports.npz {X_train_no_ports_scaled.shape}, it_test_no_ports.npz {X_test_no_ports_scaled.shape}")

    # 7. Metadata Documentation
    metadata = {
        "dataset_name": "Realistic Contiguous IT Benchmark",
        "split_methodology": "Contiguous Sequential 80/20 Block Split (Earlier 80% train, Later 20% test)",
        "limitation_disclaimer": (
            "This split divides each contiguous label block chronologically along the row order of IDS2025.xlsx. "
            "It forces out-of-burst extrapolation rather than random interpolation. However, it cannot be claimed "
            "as a true campaign-independent or zero-day evaluation because original timestamps, flow IDs, and campaign "
            "metadata are unavailable in the source file."
        ),
        "source_file": "IDS2025.xlsx",
        "raw_rows": before_len,
        "rows_after_exact_deduplication": len(df),
        "total_train_rows": len(train_indices),
        "total_test_rows": len(test_indices),
        "class_names": class_names,
        "class_to_integer": class_to_int,
        "train_class_distribution": train_counts,
        "test_class_distribution": test_counts,
        "configurations": {
            "all_features": {
                "feature_count": len(features_all),
                "features": features_all,
                "train_artifact": "it_train_all.npz",
                "test_artifact": "it_test_all.npz",
                "preprocessor_artifact": "preprocessing_all.joblib",
            },
            "no_ports": {
                "feature_count": len(features_no_ports),
                "removed_features": PORT_COLUMNS,
                "features": features_no_ports,
                "train_artifact": "it_train_no_ports.npz",
                "test_artifact": "it_test_no_ports.npz",
                "preprocessor_artifact": "preprocessing_no_ports.joblib",
            }
        },
        "contiguous_blocks": block_records,
    }

    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n[OK] Metadata and block boundary audit saved to data/processed/it_realistic/metadata.json")
    print("=== Realistic IT Data Preparation Complete ===")

if __name__ == "__main__":
    main()
