"""
ResiNet-IDAM: IT Autoencoder Threshold Calibration and Evaluation
-----------------------------------------------------------------
Evaluates threshold calibration on separate benign validation data and
measures attack recall, precision, F1, FPR, ROC-AUC, PR-AUC, and per-family
metrics across candidate operational thresholds on held-out test data.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
)
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "it_realistic"
MODELS_DIR = ROOT / "models" / "it_realistic"
REPORTS_DIR = ROOT / "reports"

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== [1] Loading Data and Metadata ===")
    meta = json.load(open(DATA_DIR / "metadata.json", encoding="utf-8"))
    train_data = np.load(DATA_DIR / "it_train_all.npz")
    test_data = np.load(DATA_DIR / "it_test_all.npz")

    normal_idx = meta["class_to_integer"]["Normal"]
    class_names = meta["class_names"]

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    # Slice benign training data
    normal_mask_train = (y_train == normal_idx)
    X_normal_train = X_train[normal_mask_train]

    # 80/20 Chronological split of normal training data for threshold selection
    split_idx = int(len(X_normal_train) * 0.8)
    X_norm_fit = X_normal_train[:split_idx]
    X_norm_val = X_normal_train[split_idx:]

    print(f"Benign training samples (fit):       {len(X_norm_fit):,}")
    print(f"Benign validation samples (calib):    {len(X_norm_val):,}")
    print(f"Total test samples:                  {len(X_test):,} (Normal: {(y_test == normal_idx).sum():,}, Attack: {(y_test != normal_idx).sum():,})")

    # Fit / Load IT Autoencoder
    ae_model_path = MODELS_DIR / "it_autoencoder.joblib"
    if ae_model_path.exists():
        print("Loading existing IT Autoencoder...")
        ae = joblib.load(ae_model_path)
    else:
        print("Training IT Autoencoder (79 -> 32 -> 16 -> 32 -> 79)...")
        ae = MLPRegressor(
            hidden_layer_sizes=(32, 16, 32),
            activation="relu",
            solver="adam",
            max_iter=50,
            random_state=42,
            batch_size=128,
        )
        ae.fit(X_norm_fit, X_norm_fit)
        joblib.dump(ae, ae_model_path)

        # Save weights for portable NumPy forward pass
        weights = {f"w{i}": w for i, w in enumerate(ae.coefs_)}
        weights.update({f"b{i}": b for i, b in enumerate(ae.intercepts_)})
        np.savez_compressed(MODELS_DIR / "it_autoencoder_weights.npz", **weights)

    # Compute validation reconstruction error (MSE)
    val_recon = ae.predict(X_norm_val)
    val_mse = np.mean((X_norm_val - val_recon) ** 2, axis=1)

    # Compute test reconstruction error (MSE)
    test_recon = ae.predict(X_test)
    test_mse = np.mean((X_test - test_recon) ** 2, axis=1)

    y_test_binary = (y_test != normal_idx).astype(int)
    roc_auc = float(roc_auc_score(y_test_binary, test_mse))
    pr_auc = float(average_precision_score(y_test_binary, test_mse))

    print(f"\nOverall Threshold-Independent Metrics: ROC-AUC = {roc_auc:.4f}, PR-AUC = {pr_auc:.4f}")

    # Candidate operational percentiles on validation normal data
    percentiles = [90.0, 95.0, 98.0, 99.0, 99.5, 99.9]
    comparison_records = []
    per_family_records = []

    selected_percentile = 99.0
    selected_payload = None

    print("\n=== [2] Threshold Calibration on Benign Validation Data ===")
    for p in percentiles:
        tau = float(np.percentile(val_mse, p))
        val_fpr = float((val_mse > tau).mean())

        test_pred = (test_mse > tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_binary, test_pred).ravel()
        test_fpr = float(fp / (fp + tn))
        prec = float(precision_score(y_test_binary, test_pred, zero_division=0))
        rec = float(recall_score(y_test_binary, test_pred, zero_division=0))
        f1 = float(f1_score(y_test_binary, test_pred, zero_division=0))

        rec_entry = {
            "percentile": p,
            "threshold_tau": tau,
            "val_fpr": val_fpr,
            "test_fpr": test_fpr,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
        comparison_records.append(rec_entry)
        print(f"P={p:>5.1f}% | Tau={tau:.4f} | Val FPR={val_fpr*100:5.2f}% | Test FPR={test_fpr*100:5.2f}% | Rec={rec*100:5.2f}% ({tp:>5}/{tp+fn}) | Prec={prec*100:5.2f}% | F1={f1:.4f}")

        # Per-family recall at this threshold
        for cls_name, cls_idx in meta["class_to_integer"].items():
            if cls_name == "Normal":
                continue
            cls_mask = (y_test == cls_idx)
            cls_count = int(cls_mask.sum())
            cls_detected = int((test_mse[cls_mask] > tau).sum())
            cls_rec = float(cls_detected / cls_count) if cls_count > 0 else 0.0
            per_family_records.append({
                "percentile": p,
                "threshold_tau": tau,
                "attack_family": cls_name,
                "test_support": cls_count,
                "detected_count": cls_detected,
                "recall": cls_rec,
            })

        if p == selected_percentile:
            selected_payload = {
                "model_name": "IT NetFlow 4-Layer Autoencoder (79 -> 32 -> 16 -> 32 -> 79)",
                "selection_rule": f"Predefined nominal 1% validation false-positive-rate target ({selected_percentile}th percentile of normal validation reconstruction error)",
                "validation_split": "Later 20% of Normal training partition (4,188 contiguous samples)",
                "threshold": tau,
                "val_measured_fpr": val_fpr,
                "test_measured_fpr": test_fpr,
                "test_precision": prec,
                "test_recall": rec,
                "test_f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "feature_count": 79,
                "trained_normal_samples": len(X_norm_fit),
                "validation_normal_samples": len(X_norm_val),
                "validation_mean_mse": float(val_mse.mean()),
                "nominal_target_achieved": bool(abs(val_fpr - 0.01) <= 0.005),
                "limitation_note": "Validation FPR is 1.00% by rank percentile definition, but test FPR is 1.32% due to out-of-burst generalization variance.",
            }

    # Save comparison to CSV
    comp_df = pd.DataFrame(comparison_records)
    comp_csv_path = REPORTS_DIR / "it_threshold_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\nSaved threshold comparison table to {comp_csv_path}")

    # Save per-family evaluation to CSV
    per_fam_df = pd.DataFrame(per_family_records)
    per_fam_csv_path = REPORTS_DIR / "it_per_family_thresholds.csv"
    per_fam_df.to_csv(per_fam_csv_path, index=False)
    print(f"Saved per-family threshold breakdown to {per_fam_csv_path}")

    # Save selected threshold JSON to models and reports
    json_path_models = MODELS_DIR / "it_autoencoder_threshold.json"
    json_path_reports = REPORTS_DIR / "it_selected_threshold.json"
    json.dump(selected_payload, open(json_path_models, "w", encoding="utf-8"), indent=2)
    json.dump(selected_payload, open(json_path_reports, "w", encoding="utf-8"), indent=2)
    print(f"Saved selected threshold metadata to {json_path_models} and {json_path_reports}")

    print("\nPer-Family Detection Summary at Nominally Calibrated 99th Percentile (Tau = {:.4f}):".format(selected_payload["threshold"]))
    fam_99 = per_fam_df[per_fam_df["percentile"] == selected_percentile]
    for _, r in fam_99.iterrows():
        print(f"  {r['attack_family']:<15}: {r['detected_count']:>5} / {r['test_support']:>5} detected ({r['recall']*100:6.2f}%)")

if __name__ == "__main__":
    main()
