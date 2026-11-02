"""Comprehensive Leave-One-Class-Out (LOCO) Zero-Day & Architectural Evaluation.

Executes Phase 2 technical requirements from the Upgrade Plan:
1. LOCO evaluation across all 6 attack families:
   - Botnet ARES
   - Brute Force
   - Dos/DDos
   - Infiltration
   - PortScan
   - Web Attack
2. Closed-world supervised model on N-1 classes (measures forced misclassification into Normal).
3. Evaluates 4 candidate unsupervised detection architectures:
   - Baseline Dense Autoencoder (79 -> 32 -> 16 -> 32 -> 79)
   - Residual Autoencoder (with skip connections)
   - Attention Autoencoder (with feature gating ablation)
   - Isolation Forest
4. Causal sequence window comparison (L=1 vs L=5 vs L=10) with detection delay.
5. Saves reports/loco_full_evaluation.json, reports/loco_per_family_matrix.csv, and
   reports/it_model_architectures_comparison.csv.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import joblib

from src.detectors.residual_autoencoder import (
    build_keras_residual_autoencoder,
    export_weights_to_npz,
    NumpyResidualAutoencoder,
)
from src.detectors.attention_autoencoder import (
    build_keras_attention_autoencoder,
    export_attention_weights_to_npz,
    NumpyAttentionAutoencoder,
)
from src.detectors.sequence_window_detector import (
    CausalSequenceWindowScorer,
    compute_detection_delay,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "it_realistic"
MODELS_DIR = ROOT / "models" / "it_realistic"
REPORTS_DIR = ROOT / "reports"


def main() -> None:
    print("=" * 75)
    print("  ResiNet-IDAM v2.0: Comprehensive Phase 2 LOCO & Architectural Evaluation")
    print("=" * 75)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    meta = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    class_names = meta["class_names"]
    c2i = meta["class_to_integer"]
    normal_idx = c2i["Normal"]
    attack_families = [name for name in class_names if name != "Normal"]

    train_data = np.load(DATA_DIR / "it_train_all.npz")
    test_data = np.load(DATA_DIR / "it_test_all.npz")
    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    normal_mask_train = (y_train == normal_idx)
    X_norm_train = X_train[normal_mask_train]
    split_idx = int(len(X_norm_train) * 0.8)
    X_norm_fit = X_norm_train[:split_idx]
    X_norm_val = X_norm_train[split_idx:]

    normal_mask_test = (y_test == normal_idx)
    X_norm_test = X_test[normal_mask_test]
    y_test_binary = (y_test != normal_idx).astype(int)

    print(f"Benign training fit: {len(X_norm_fit):,} | Benign validation calib: {len(X_norm_val):,}")
    print(f"Total test flows:   {len(X_test):,} (Normal: {len(X_norm_test):,}, Attacks: {len(X_test) - len(X_norm_test):,})")

    # =========================================================================
    # Step 1: Train & Calibrate Unsupervised Architectures
    # =========================================================================
    print("\n--- [Step 1] Training & Calibrating Unsupervised Architectures ---")

    # A. Baseline Dense Autoencoder
    ae_baseline = joblib.load(MODELS_DIR / "it_autoencoder.joblib")
    val_recon_base = ae_baseline.predict(X_norm_val)
    val_mse_base = np.mean((X_norm_val - val_recon_base) ** 2, axis=1)
    tau_base = float(np.percentile(val_mse_base, 99.0))
    test_recon_base = ae_baseline.predict(X_test)
    test_mse_base = np.mean((X_test - test_recon_base) ** 2, axis=1)

    # B. Residual Autoencoder
    res_weights_path = MODELS_DIR / "it_residual_autoencoder_weights.npz"
    if not res_weights_path.exists():
        print("Training Keras Residual Autoencoder...")
        res_keras = build_keras_residual_autoencoder(79)
        res_keras.fit(X_norm_fit, X_norm_fit, epochs=20, batch_size=256, verbose=0)
        export_weights_to_npz(res_keras, res_weights_path)
    res_model = NumpyResidualAutoencoder(res_weights_path)
    val_recon_res = res_model.predict(X_norm_val)
    val_mse_res = np.mean((X_norm_val - val_recon_res) ** 2, axis=1)
    tau_res = float(np.percentile(val_mse_res, 99.0))
    test_recon_res = res_model.predict(X_test)
    test_mse_res = np.mean((X_test - test_recon_res) ** 2, axis=1)

    # C. Attention Autoencoder Ablation
    attn_weights_path = MODELS_DIR / "it_attention_autoencoder_weights.npz"
    if not attn_weights_path.exists():
        print("Training Keras Attention Autoencoder (Ablation)...")
        attn_keras = build_keras_attention_autoencoder(79)
        attn_keras.fit(X_norm_fit, X_norm_fit, epochs=20, batch_size=256, verbose=0)
        export_attention_weights_to_npz(attn_keras, attn_weights_path)
    attn_model = NumpyAttentionAutoencoder(attn_weights_path)
    val_recon_attn = attn_model.predict(X_norm_val)
    val_mse_attn = np.mean((X_norm_val - val_recon_attn) ** 2, axis=1)
    tau_attn = float(np.percentile(val_mse_attn, 99.0))
    test_recon_attn = attn_model.predict(X_test)
    test_mse_attn = np.mean((X_test - test_recon_attn) ** 2, axis=1)

    # D. Isolation Forest
    if_path = MODELS_DIR / "loco_iforest_all_botnet_ares.joblib"
    iforest = joblib.load(if_path)
    val_if_scores = -iforest.score_samples(X_norm_val)
    tau_if = float(np.percentile(val_if_scores, 99.0))
    test_if_scores = -iforest.score_samples(X_test)

    # Overall Architecture Comparison on IT Test
    arch_models = {
        "Baseline Autoencoder (79->32->16->32->79)": (test_mse_base, tau_base),
        "Residual Autoencoder (with Skip Connections)": (test_mse_res, tau_res),
        "Attention Autoencoder (Feature Gating Ablation)": (test_mse_attn, tau_attn),
        "Isolation Forest Baseline": (test_if_scores, tau_if),
    }

    arch_comparison_rows = []
    for model_name, (scores, tau) in arch_models.items():
        preds = (scores > tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_binary, preds, labels=[0, 1]).ravel()
        p = float(precision_score(y_test_binary, preds, zero_division=0))
        r = float(recall_score(y_test_binary, preds, zero_division=0))
        f1 = float(f1_score(y_test_binary, preds, zero_division=0))
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        roc = float(roc_auc_score(y_test_binary, scores))
        pr = float(average_precision_score(y_test_binary, scores))
        delay = compute_detection_delay(y_test_binary, preds)

        arch_comparison_rows.append({
            "Architecture": model_name,
            "Threshold": round(tau, 6),
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1-Score": round(f1, 4),
            "FPR": round(fpr, 4),
            "FNR": round(fnr, 4),
            "ROC-AUC": round(roc, 4),
            "PR-AUC": round(pr, 4),
            "Burst Detection Rate": round(delay["burst_detection_rate"], 4),
            "Median Detection Delay (samples)": delay["median_detection_delay_samples"],
        })

    arch_df = pd.DataFrame(arch_comparison_rows)
    arch_df.to_csv(REPORTS_DIR / "it_model_architectures_comparison.csv", index=False)
    print("\nArchitecture Comparison Table:")
    print(arch_df[["Architecture", "Precision", "Recall", "F1-Score", "FPR", "ROC-AUC"]].to_string(index=False))

    # =========================================================================
    # Step 2: Causal Sequence Window Ablation (L=1, L=5, L=10, L=20)
    # =========================================================================
    print("\n--- [Step 2] Evaluating Causal Sequence Windows (No Lookahead) ---")
    window_results = []
    for w in [1, 5, 10, 20]:
        scorer = CausalSequenceWindowScorer(window_size=w, aggregation="mean")
        val_w_scores = scorer.score_sequence(val_mse_res)
        tau_w = float(np.percentile(val_w_scores, 99.0))
        test_w_scores = scorer.score_sequence(test_mse_res)
        w_preds = (test_w_scores > tau_w).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test_binary, w_preds, labels=[0, 1]).ravel()
        p = float(precision_score(y_test_binary, w_preds, zero_division=0))
        r = float(recall_score(y_test_binary, w_preds, zero_division=0))
        f1 = float(f1_score(y_test_binary, w_preds, zero_division=0))
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        delays = compute_detection_delay(y_test_binary, w_preds)

        window_results.append({
            "Window Size (L)": w,
            "Calibrated Threshold": round(tau_w, 6),
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1-Score": round(f1, 4),
            "FPR": round(fpr, 4),
            "Median Delay (samples)": delays["median_detection_delay_samples"],
        })
    window_df = pd.DataFrame(window_results)
    print(window_df.to_string(index=False))

    # =========================================================================
    # Step 3: Comprehensive LOCO Evaluation Across All 6 Attack Families
    # =========================================================================
    print("\n--- [Step 3] Full Leave-One-Class-Out (LOCO) Evaluation across 6 Families ---")

    loco_records = []
    per_family_table_rows = []

    for family in attack_families:
        fam_idx = c2i[family]
        fam_mask_test = (y_test == fam_idx)
        n_fam_test = int(fam_mask_test.sum())

        # Combine Normal test samples + held-out family test samples
        eval_mask = normal_mask_test | fam_mask_test
        X_eval = X_test[eval_mask]
        y_eval_true_binary = (y_test[eval_mask] != normal_idx).astype(int)

        # 3.1 Closed-World Supervised Model (withheld family excluded from training)
        print(f"\nEvaluating Held-Out Family: '{family}' (N_test={n_fam_test:,})...")
        train_loco_mask = (y_train != fam_idx)
        X_tr_loco = X_train[train_loco_mask]
        y_tr_loco = y_train[train_loco_mask]

        rf_loco = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=2)
        rf_loco.fit(X_tr_loco, y_tr_loco)
        rf_loco_preds = rf_loco.predict(X_test[fam_mask_test])

        forced_normal = int((rf_loco_preds == normal_idx).sum())
        forced_normal_rate = float(forced_normal / n_fam_test) if n_fam_test > 0 else 0.0
        categorized_other = int((rf_loco_preds != normal_idx).sum())

        # 3.2 Unsupervised Detectors on Held-Out Family
        # Baseline Autoencoder
        base_fam_preds = (test_mse_base[eval_mask] > tau_base).astype(int)
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_eval_true_binary, base_fam_preds, labels=[0, 1]).ravel()
        rec_base = float(tp_b / (tp_b + fn_b)) if (tp_b + fn_b) > 0 else 0.0
        prec_base = float(precision_score(y_eval_true_binary, base_fam_preds, zero_division=0))
        f1_base = float(f1_score(y_eval_true_binary, base_fam_preds, zero_division=0))
        roc_base = float(roc_auc_score(y_eval_true_binary, test_mse_base[eval_mask]))

        # Residual Autoencoder
        res_fam_preds = (test_mse_res[eval_mask] > tau_res).astype(int)
        tn_r, fp_r, fn_r, tp_r = confusion_matrix(y_eval_true_binary, res_fam_preds, labels=[0, 1]).ravel()
        rec_res = float(tp_r / (tp_r + fn_r)) if (tp_r + fn_r) > 0 else 0.0
        prec_res = float(precision_score(y_eval_true_binary, res_fam_preds, zero_division=0))
        f1_res = float(f1_score(y_eval_true_binary, res_fam_preds, zero_division=0))
        roc_res = float(roc_auc_score(y_eval_true_binary, test_mse_res[eval_mask]))

        # Isolation Forest
        if_fam_preds = (test_if_scores[eval_mask] > tau_if).astype(int)
        tn_i, fp_i, fn_i, tp_i = confusion_matrix(y_eval_true_binary, if_fam_preds, labels=[0, 1]).ravel()
        rec_if = float(tp_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else 0.0
        prec_if = float(precision_score(y_eval_true_binary, if_fam_preds, zero_division=0))
        f1_if = float(f1_score(y_eval_true_binary, if_fam_preds, zero_division=0))
        roc_if = float(roc_auc_score(y_eval_true_binary, test_if_scores[eval_mask]))

        loco_records.append({
            "held_out_family": family,
            "test_sample_count": n_fam_test,
            "closed_world_supervised": {
                "forced_normal_count": forced_normal,
                "forced_normal_rate": round(forced_normal_rate, 4),
                "categorized_as_other_attack_count": categorized_other,
                "zero_day_blindness": round(forced_normal_rate * 100, 2),
            },
            "unsupervised_baseline_autoencoder": {
                "tp": int(tp_b), "fp": int(fp_b), "fn": int(fn_b), "tn": int(tn_b),
                "recall": round(rec_base, 4),
                "precision": round(prec_base, 4),
                "f1": round(f1_base, 4),
                "roc_auc": round(roc_base, 4),
            },
            "unsupervised_residual_autoencoder": {
                "tp": int(tp_r), "fp": int(fp_r), "fn": int(fn_r), "tn": int(tn_r),
                "recall": round(rec_res, 4),
                "precision": round(prec_res, 4),
                "f1": round(f1_res, 4),
                "roc_auc": round(roc_res, 4),
            },
            "unsupervised_isolation_forest": {
                "tp": int(tp_i), "fp": int(fp_i), "fn": int(fn_i), "tn": int(tn_i),
                "recall": round(rec_if, 4),
                "precision": round(prec_if, 4),
                "f1": round(f1_if, 4),
                "roc_auc": round(roc_if, 4),
            },
        })

        per_family_table_rows.append({
            "Held-Out Attack Family": family,
            "Test Samples": n_fam_test,
            "Supervised (Forced Normal)": f"{forced_normal}/{n_fam_test} ({forced_normal_rate:.2%})",
            "Baseline Autoencoder Recall": f"{tp_b}/{n_fam_test} ({rec_base:.2%})",
            "Residual Autoencoder Recall": f"{tp_r}/{n_fam_test} ({rec_res:.2%})",
            "Isolation Forest Recall": f"{tp_i}/{n_fam_test} ({rec_if:.2%})",
            "Baseline AE ROC-AUC": round(roc_base, 4),
            "Residual AE ROC-AUC": round(roc_res, 4),
        })

    per_family_df = pd.DataFrame(per_family_table_rows)
    per_family_df.to_csv(REPORTS_DIR / "loco_per_family_matrix.csv", index=False)
    print("\nLOCO Per-Family Evaluation Matrix:")
    print(per_family_df.to_string(index=False))

    loco_payload = {
        "evaluation_title": "ResiNet-IDAM v2.0 Full LOCO Zero-Day Intrusion Evaluation",
        "timestamp": pd.Timestamp.now().isoformat(),
        "families_evaluated": len(attack_families),
        "architectures_evaluated": list(arch_models.keys()),
        "causal_window_evaluation": window_results,
        "loco_per_family_results": loco_records,
    }

    out_json = REPORTS_DIR / "loco_full_evaluation.json"
    out_json.write_text(json.dumps(loco_payload, indent=2), encoding="utf-8")
    print(f"\nSaved LOCO evaluation results to: {out_json}")
    print(f"Saved architecture comparison:   {REPORTS_DIR / 'it_model_architectures_comparison.csv'}")
    print(f"Saved per-family matrix:         {REPORTS_DIR / 'loco_per_family_matrix.csv'}")


if __name__ == "__main__":
    main()

