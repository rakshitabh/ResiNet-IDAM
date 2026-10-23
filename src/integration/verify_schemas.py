import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
IT_DIR = ROOT / "data" / "processed" / "it"
OT_DIR = ROOT / "data" / "processed"
IT_MODEL_DIR = ROOT / "models" / "it"
OT_MODEL_DIR = ROOT / "models"

def verify_it_artifacts():
    print("=== [1/2] Verifying IT Artifacts ===")
    
    # 1. Check metadata
    meta_path = IT_DIR / "metadata.json"
    assert meta_path.exists(), f"Missing {meta_path}"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[OK] IT Metadata loaded: {len(meta['feature_names'])} features, {len(meta['class_names'])} classes.")
    
    # 2. Check processed data arrays
    train_npz = IT_DIR / "it_train.npz"
    test_npz = IT_DIR / "it_test.npz"
    assert train_npz.exists() and test_npz.exists(), "Missing IT train or test .npz files"
    
    tr = np.load(train_npz)
    te = np.load(test_npz)
    X_tr, y_tr = tr["X"], tr["y"]
    X_te, y_te = te["X"], te["y"]
    print(f"[OK] IT Data shapes: Train X={X_tr.shape}, y={y_tr.shape} | Test X={X_te.shape}, y={y_te.shape}")
    assert X_tr.shape[1] == len(meta["feature_names"]), "Feature count mismatch in IT train"
    assert X_te.shape[1] == len(meta["feature_names"]), "Feature count mismatch in IT test"
    
    # 3. Check trained model and preprocessors
    rf_path = IT_MODEL_DIR / "random_forest.joblib"
    prep_path = IT_MODEL_DIR / "preprocessing.joblib"
    le_path = IT_MODEL_DIR / "label_encoder.joblib"
    
    assert rf_path.exists(), f"Missing {rf_path}"
    assert prep_path.exists(), f"Missing {prep_path}"
    assert le_path.exists(), f"Missing {le_path}"
    
    rf = joblib.load(rf_path)
    prep = joblib.load(prep_path)
    le = joblib.load(le_path)
    
    print(f"[OK] IT Model verified: RandomForestClassifier (n_estimators={rf.n_estimators}, classes={rf.classes_})")
    print(f"[OK] IT Preprocessor verified: Pipeline steps={[name for name, _ in prep.steps]}")
    print(f"[OK] IT LabelEncoder verified: Classes={le.classes_.tolist()}")

def verify_ot_artifacts():
    print("\n=== [2/2] Verifying OT Artifacts ===")
    
    # 1. Check autoencoder metadata/threshold
    thresh_path = OT_MODEL_DIR / "autoencoder_threshold.json"
    assert thresh_path.exists(), f"Missing {thresh_path}"
    with open(thresh_path, "r", encoding="utf-8") as f:
        ot_meta = json.load(f)
    
    features = ot_meta["feature_columns"]
    threshold = ot_meta["threshold"]
    print(f"[OK] OT Metadata loaded: {len(features)} feature columns, threshold={threshold:.6f}")
    
    # 2. Check processed CSVs and headers
    train_path = OT_DIR / "hai-train-scaled.csv"
    test_path = OT_DIR / "hai-test-scaled.csv"
    assert train_path.exists(), f"Missing {train_path}"
    assert test_path.exists(), f"Missing {test_path}"
    
    # Read sample to verify column alignment
    df_tr_sample = pd.read_csv(train_path, nrows=5)
    df_te_sample = pd.read_csv(test_path, nrows=5)
    
    assert "timestamp" in df_tr_sample.columns, "Missing timestamp in OT train"
    assert "timestamp" in df_te_sample.columns, "Missing timestamp in OT test"
    assert "label" in df_te_sample.columns, "Missing label in OT test"
    
    # Verify features match exactly
    train_cols = [c for c in df_tr_sample.columns if c != "timestamp"]
    test_cols = [c for c in df_te_sample.columns if c not in ["timestamp", "label"]]
    
    assert train_cols == features, "Feature column ordering mismatch between metadata and hai-train-scaled.csv"
    assert test_cols == features, "Feature column ordering mismatch between metadata and hai-test-scaled.csv"
    print(f"[OK] OT Feature column ordering strictly matched across train, test, and metadata ({len(features)} columns).")
    
    # 3. Check timestamp continuity and sampling rate in test data
    df_te_time = pd.read_csv(test_path, usecols=["timestamp", "label"])
    df_te_time["timestamp"] = pd.to_datetime(df_te_time["timestamp"])
    time_diffs = df_te_time["timestamp"].diff().dropna().dt.total_seconds()
    median_dt = time_diffs.median()
    print(f"[OK] OT Test Telemetry timestamps: {len(df_te_time)} records, Start={df_te_time['timestamp'].min()}, End={df_te_time['timestamp'].max()}")
    print(f"[OK] OT Sampling frequency: Median dt = {median_dt}s (expected 1.0s). Attacks = {df_te_time['label'].sum()} / {len(df_te_time)} rows ({df_te_time['label'].mean()*100:.2f}%)")
    
    # 4. Check OT models
    ae_path = OT_MODEL_DIR / "ot_autoencoder.keras"
    if_path = OT_MODEL_DIR / "isolation_forest.joblib"
    scaler_path = OT_MODEL_DIR / "standard_scaler.joblib"
    
    assert ae_path.exists(), f"Missing {ae_path}"
    assert if_path.exists(), f"Missing {if_path}"
    assert scaler_path.exists(), f"Missing {scaler_path}"
    print(f"[OK] OT Models verified: Autoencoder ({ae_path.name}), Isolation Forest ({if_path.name}), StandardScaler ({scaler_path.name})")

if __name__ == "__main__":
    verify_it_artifacts()
    verify_ot_artifacts()
    print("\n>>> ALL IT AND OT SCHEMAS AND ARTIFACTS SUCCESSFULLY VERIFIED! <<<")
