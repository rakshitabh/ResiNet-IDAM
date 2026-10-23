"""
ResiNet-IDAM: Score Calibration Module
Extracts and calibrates scalar anomaly scores:
- IT scalar attack probability: p_attack(t) = 1 - P_IT(Normal, t)
- OT normalized reconstruction score: S_OT(t) = sigmoid(gamma * (MSE(t) - theta))
- Zero-Trust context factor: ZT(t)

All calibrations and parameter optimizations (e.g. gamma) are strictly fit
on the VALIDATION partition (scenarios_val.csv), leaving the TEST partition
(scenarios_test.csv) completely untouched.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
IT_DIR = ROOT / "data" / "processed" / "it"
OT_DIR = ROOT / "data" / "processed"
INTEG_DIR = ROOT / "data" / "processed" / "integration"
MODELS_DIR = ROOT / "models"

def main():
    print("=== [Step 3] Calibrating IT and OT Detection Scores ===")

    # 1. Load models
    print("Loading IT and OT models...")
    rf_model = joblib.load(MODELS_DIR / "it" / "random_forest.joblib")
    with open(IT_DIR / "metadata.json", "r", encoding="utf-8") as f:
        it_meta = json.load(f)
    normal_class_idx = it_meta["class_to_integer"]["Normal"]

    ae_model = keras.models.load_model(MODELS_DIR / "ot_autoencoder.keras")
    with open(MODELS_DIR / "autoencoder_threshold.json", "r", encoding="utf-8") as f:
        ot_meta = json.load(f)
    ot_features = ot_meta["feature_columns"]
    ot_threshold = float(ot_meta["threshold"])
    print(f"  OT Autoencoder threshold: {ot_threshold:.6f}")

    # 2. Load IT test features and OT test features
    it_test_npz = np.load(IT_DIR / "it_test.npz")
    X_it_all = it_test_npz["X"]

    ot_test_df = pd.read_csv(OT_DIR / "hai-test-scaled.csv")
    X_ot_all = ot_test_df[ot_features].to_numpy(dtype=np.float32)

    # 3. Compute OT Autoencoder reconstruction errors for all records
    print("Computing OT Autoencoder reconstruction errors (batch_size=512)...")
    reconstructed = ae_model.predict(X_ot_all, batch_size=512, verbose=0)
    ot_mse_all = np.mean(np.square(X_ot_all - reconstructed), axis=1)

    # 4. Load Scenarios partitions
    val_scen = pd.read_csv(INTEG_DIR / "scenarios_val.csv")
    test_scen = pd.read_csv(INTEG_DIR / "scenarios_test.csv")

    # Helper function to extract predictions for a partition
    def extract_scores(scen_df):
        # Extract IT features matching each row's assigned flow
        it_indices = scen_df["it_flow_idx"].values
        X_it_partition = X_it_all[it_indices]

        # Predict class probabilities
        probs = rf_model.predict_proba(X_it_partition)
        # Scalar IT attack score: 1 - P(Normal)
        p_normal = probs[:, normal_class_idx]
        p_attack = (1.0 - p_normal).astype(np.float32)

        # Extract corresponding OT MSE
        ot_indices = scen_df["ot_record_idx"].values
        ot_mse = ot_mse_all[ot_indices].astype(np.float32)

        zt_context = scen_df["zt_context"].values.astype(np.float32)

        return p_attack, ot_mse, zt_context

    print("\nExtracting scores for Validation partition...")
    p_attack_val, ot_mse_val, zt_val = extract_scores(val_scen)

    print("Extracting scores for Test partition...")
    p_attack_test, ot_mse_test, zt_test = extract_scores(test_scen)

    # 5. Fit sigmoid scaling temperature gamma strictly on VALIDATION partition
    # We want sigmoid(gamma * (MSE - theta)) to produce smooth separation around threshold
    # Optimize gamma to maximize binary cross-entropy alignment with validation ot_label
    y_ot_val = val_scen["ot_label"].values
    
    def loss(gamma):
        # Prevent overflow
        diff = np.clip(gamma * (ot_mse_val - ot_threshold), -20, 20)
        s_pred = 1.0 / (1.0 + np.exp(-diff))
        # Log loss with epsilon clipping
        eps = 1e-7
        s_pred = np.clip(s_pred, eps, 1.0 - eps)
        bce = -np.mean(y_ot_val * np.log(s_pred) + (1.0 - y_ot_val) * np.log(1.0 - s_pred))
        return bce

    res = minimize_scalar(loss, bounds=(0.5, 30.0), method="bounded")
    best_gamma = float(res.x)
    print(f"\n[OK] Calibrated sigmoid temperature gamma on validation set: {best_gamma:.4f}")

    # Compute calibrated S_OT
    diff_val = np.clip(best_gamma * (ot_mse_val - ot_threshold), -20, 20)
    S_ot_val = (1.0 / (1.0 + np.exp(-diff_val))).astype(np.float32)

    diff_test = np.clip(best_gamma * (ot_mse_test - ot_threshold), -20, 20)
    S_ot_test = (1.0 / (1.0 + np.exp(-diff_test))).astype(np.float32)

    # 6. Save calibrated scores
    np.savez_compressed(
        INTEG_DIR / "scores_val.npz",
        p_attack=p_attack_val,
        S_ot=S_ot_val,
        zt=zt_val,
        ot_mse=ot_mse_val,
        ot_label=y_ot_val,
        is_cyber_physical=val_scen["is_cyber_physical"].values,
    )

    np.savez_compressed(
        INTEG_DIR / "scores_test.npz",
        p_attack=p_attack_test,
        S_ot=S_ot_test,
        zt=zt_test,
        ot_mse=ot_mse_test,
        ot_label=test_scen["ot_label"].values,
        is_cyber_physical=test_scen["is_cyber_physical"].values,
    )

    calibration_meta = {
        "ot_threshold": ot_threshold,
        "calibrated_gamma": best_gamma,
        "validation_records": len(val_scen),
        "test_records": len(test_scen),
        "validation_bce_loss": float(res.fun),
    }
    with open(MODELS_DIR / "integration_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration_meta, f, indent=2)

    print(f"[OK] Calibrated scores saved to scores_val.npz and scores_test.npz")
    print(f"[OK] Calibration parameters saved to models/integration_calibration.json")
    print("\nSummary of Calibrated Scores on Test Partition:")
    print(f"  p_attack: mean={p_attack_test.mean():.4f}, min={p_attack_test.min():.4f}, max={p_attack_test.max():.4f}")
    print(f"  S_ot:     mean={S_ot_test.mean():.4f}, min={S_ot_test.min():.4f}, max={S_ot_test.max():.4f}")
    print(f"  ZT:       mean={zt_test.mean():.4f}, min={zt_test.min():.4f}, max={zt_test.max():.4f}")

if __name__ == "__main__":
    main()
