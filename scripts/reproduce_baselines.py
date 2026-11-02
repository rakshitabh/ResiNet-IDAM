"""Master Reproducible Baseline Verification Script for ResiNet-IDAM v2.0.

Evaluates all four IT and OT baseline models on frozen test datasets,
verifies exact metric reproduction against the Upgrade Plan,
and writes reports/reproducible_baseline_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import joblib

ROOT = Path(__file__).resolve().parents[1]
IT_DATA_DIR = ROOT / "data" / "processed" / "it_realistic"
IT_MODEL_DIR = ROOT / "models" / "it_realistic"
OT_DATA_PATH = ROOT / "data" / "processed" / "hai-test-scaled.csv"
OT_MODEL_DIR = ROOT / "models"
OUTPUT_PATH = ROOT / "reports" / "reproducible_baseline_results.json"


def evaluate_it_rf() -> dict:
    meta = json.loads((IT_DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    test_data = np.load(IT_DATA_DIR / "it_test_all.npz")
    X_test, y_test = test_data["X"], test_data["y"]

    normal_idx = meta["class_to_integer"]["Normal"]
    class_names = meta["class_names"]

    rf = joblib.load(IT_MODEL_DIR / "random_forest_all.joblib")
    y_pred = rf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    # Per-family recall
    per_family = {}
    for idx, name in enumerate(class_names):
        mask = (y_test == idx)
        total = int(mask.sum())
        detected = int((y_pred[mask] == idx).sum()) if idx == normal_idx else int((y_pred[mask] != normal_idx).sum())
        rec = float(detected / total) if total > 0 else 0.0
        per_family[name] = {
            "total_samples": total,
            "detected": detected,
            "recall": round(rec, 4),
        }

    return {
        "model_name": "Random Forest (Supervised 79-feature)",
        "paradigm": "Supervised Multi-Class Classification",
        "dataset": "IDS2025 Realistic IT Test",
        "total_samples": len(y_test),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_family": per_family,
    }


def evaluate_it_ae() -> dict:
    meta = json.loads((IT_DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    test_data = np.load(IT_DATA_DIR / "it_test_all.npz")
    X_test, y_test = test_data["X"], test_data["y"]

    normal_idx = meta["class_to_integer"]["Normal"]
    class_names = meta["class_names"]
    y_test_binary = (y_test != normal_idx).astype(int)

    ae = joblib.load(IT_MODEL_DIR / "it_autoencoder.joblib")
    thresh_meta = json.loads((IT_MODEL_DIR / "it_autoencoder_threshold.json").read_text(encoding="utf-8"))
    tau = float(thresh_meta["threshold"])

    recon = ae.predict(X_test)
    mse = np.mean((X_test - recon) ** 2, axis=1)
    y_pred = (mse > tau).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred, labels=[0, 1]).ravel()
    prec = float(precision_score(y_test_binary, y_pred, zero_division=0))
    rec = float(recall_score(y_test_binary, y_pred, zero_division=0))
    f1 = float(f1_score(y_test_binary, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    roc_auc = float(roc_auc_score(y_test_binary, mse))
    pr_auc = float(average_precision_score(y_test_binary, mse))

    per_family = {}
    for idx, name in enumerate(class_names):
        if idx == normal_idx:
            continue
        mask = (y_test == idx)
        total = int(mask.sum())
        detected = int(y_pred[mask].sum())
        per_family[name] = {
            "total_samples": total,
            "detected": detected,
            "recall": round(detected / total, 4) if total > 0 else 0.0,
        }

    return {
        "model_name": "IT Autoencoder (79 -> 32 -> 16 -> 32 -> 79)",
        "paradigm": "Unsupervised Novelty Detection",
        "dataset": "IDS2025 Realistic IT Test",
        "threshold": round(tau, 6),
        "total_samples": len(y_test),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_family": per_family,
    }


def evaluate_ot_ae() -> dict:
    df = pd.read_csv(OT_DATA_PATH)
    meta = json.loads((OT_MODEL_DIR / "autoencoder_threshold.json").read_text(encoding="utf-8"))
    features = meta["feature_columns"]
    tau = float(meta["threshold"])

    X = df[features].values.astype(np.float32)
    y = df["label"].values.astype(int)

    weights = np.load(OT_MODEL_DIR / "ot_autoencoder_weights.npz")
    w0, b0 = weights["w0"], weights["b0"]
    w1, b1 = weights["w1"], weights["b1"]
    w2, b2 = weights["w2"], weights["b2"]
    w3, b3 = weights["w3"], weights["b3"]

    h0 = np.maximum(0, X @ w0 + b0)
    h1 = np.maximum(0, h0 @ w1 + b1)
    h2 = np.maximum(0, h1 @ w2 + b2)
    recon = h2 @ w3 + b3
    mse = np.mean((X - recon) ** 2, axis=1)

    y_pred = (mse > tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    roc_auc = float(roc_auc_score(y, mse))

    return {
        "model_name": "OT Autoencoder (66 -> 32 -> 16 -> 32 -> 66)",
        "paradigm": "Unsupervised Industrial Anomaly Detection",
        "dataset": "HAI 21.03/22.04 Test",
        "threshold": round(tau, 6),
        "total_samples": len(y),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_ot_if() -> dict:
    df = pd.read_csv(OT_DATA_PATH)
    features = [c for c in df.columns if c not in ("timestamp", "label")]
    X = df[features].values
    y = df["label"].values.astype(int)

    iforest = joblib.load(OT_MODEL_DIR / "isolation_forest.joblib")
    preds = iforest.predict(X)
    y_pred = (preds == -1).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "model_name": "OT Isolation Forest",
        "paradigm": "Unsupervised Density Outlier",
        "dataset": "HAI 21.03/22.04 Test",
        "total_samples": len(y),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "false_positive_count": int(fp),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    print("=" * 70)
    print("  ResiNet-IDAM v2.0: Master Reproducible Baseline Benchmark")
    print("=" * 70)

    print("\n[1/4] Evaluating IT Supervised Random Forest...")
    it_rf = evaluate_it_rf()
    print(f"      Accuracy: {it_rf['accuracy'] * 100:.2f}% | Macro F1: {it_rf['macro_f1'] * 100:.2f}%")
    print(f"      Botnet ARES Recall: {it_rf['per_family']['Botnet ARES']['recall'] * 100:.2f}%")

    print("\n[2/4] Evaluating IT Unsupervised Autoencoder...")
    it_ae = evaluate_it_ae()
    print(f"      Precision: {it_ae['precision'] * 100:.2f}% | Recall: {it_ae['recall'] * 100:.2f}% | F1: {it_ae['f1'] * 100:.2f}% | ROC-AUC: {it_ae['roc_auc'] * 100:.2f}%")
    print(f"      Botnet ARES Recall: {it_ae['per_family']['Botnet ARES']['recall'] * 100:.2f}% ({it_ae['per_family']['Botnet ARES']['detected']}/{it_ae['per_family']['Botnet ARES']['total_samples']})")

    print("\n[3/4] Evaluating OT Unsupervised Autoencoder...")
    ot_ae = evaluate_ot_ae()
    print(f"      Precision: {ot_ae['precision'] * 100:.2f}% | Recall: {ot_ae['recall'] * 100:.2f}% | F1: {ot_ae['f1'] * 100:.2f}% | FPR: {ot_ae['fpr'] * 100:.2f}%")
    print(f"      CM: TN={ot_ae['confusion_matrix']['tn']}, FP={ot_ae['confusion_matrix']['fp']}, FN={ot_ae['confusion_matrix']['fn']}, TP={ot_ae['confusion_matrix']['tp']}")

    print("\n[4/4] Evaluating OT Unsupervised Isolation Forest...")
    ot_if = evaluate_ot_if()
    print(f"      Precision: {ot_if['precision'] * 100:.2f}% | Recall: {ot_if['recall'] * 100:.2f}% | F1: {ot_if['f1'] * 100:.2f}% | FP: {ot_if['false_positive_count']}")

    results = {
        "benchmark_timestamp": pd.Timestamp.now().isoformat(),
        "verification_against_plan": {
            "IT_Random_Forest": {
                "expected": {"accuracy": 0.9958, "macro_f1": 0.9682},
                "measured": {"accuracy": it_rf["accuracy"], "macro_f1": it_rf["macro_f1"]},
                "verified": abs(it_rf["accuracy"] - 0.9958) <= 0.001,
            },
            "IT_Autoencoder": {
                "expected": {"precision": 0.9558, "recall": 0.1155, "f1": 0.2061, "botnet_ares_recall": 0.0240},
                "measured": {
                    "precision": it_ae["precision"],
                    "recall": it_ae["recall"],
                    "f1": it_ae["f1"],
                    "botnet_ares_recall": it_ae["per_family"]["Botnet ARES"]["recall"],
                },
                "verified": abs(it_ae["recall"] - 0.1155) <= 0.001,
            },
            "OT_Autoencoder": {
                "expected": {"precision": 0.3492, "recall": 0.4639, "f1": 0.3984},
                "measured": {"precision": ot_ae["precision"], "recall": ot_ae["recall"], "f1": ot_ae["f1"]},
                "verified": abs(ot_ae["f1"] - 0.3984) <= 0.001,
            },
            "OT_Isolation_Forest": {
                "expected": {"precision": 0.1049, "recall": 0.6880, "f1": 0.1820, "fp": 17506},
                "measured": {
                    "precision": ot_if["precision"],
                    "recall": ot_if["recall"],
                    "f1": ot_if["f1"],
                    "fp": ot_if["false_positive_count"],
                },
                "verified": abs(ot_if["f1"] - 0.1820) <= 0.001 and ot_if["false_positive_count"] == 17506,
            },
        },
        "models": {
            "IT_Random_Forest": it_rf,
            "IT_Autoencoder": it_ae,
            "OT_Autoencoder": ot_ae,
            "OT_Isolation_Forest": ot_if,
        },
    }

    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved reproducible baseline results: {OUTPUT_PATH}")

    all_verified = all(
        item["verified"] for item in results["verification_against_plan"].values()
    )
    print(f"\nALL BASELINES VERIFIED AGAINST PLAN: {all_verified}")
    assert all_verified, "Baseline reproduction failed to match target metrics within tolerance!"


if __name__ == "__main__":
    main()

