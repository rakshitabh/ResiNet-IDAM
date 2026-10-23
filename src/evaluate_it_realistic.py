"""
ResiNet-IDAM: Realistic IT Benchmark Evaluation Script
------------------------------------------------------
Evaluates the baseline Random Forest on the contiguous temporal-proxy split
under two independent feature configurations:
  Configuration A: All 79 numeric network features.
  Configuration B: 77 numeric features (Source and Destination Ports removed).

Model Settings (Identical to V1.0 Baseline):
  RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=2)

Evaluation Metrics:
- Overall Accuracy, Macro F1, Weighted F1
- Per-class Precision, Recall, F1, and Support
- 7x7 Multi-Class Confusion Matrix
- Explicit False-Positive Rate (FPR = FP / (FP + TN), Normal=Negative, Attacks=Positive)
- Diagnostic feature-vector overlap analysis (including the 824 exact matches in Config B)
- Direct comparison with frozen random-split baseline (99.56% Acc, 96.76% Macro F1)

Outputs saved separately:
- data/reports/it_realistic_evaluation_report.txt
- models/it_realistic/random_forest_all.joblib
- models/it_realistic/random_forest_no_ports.joblib
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "it_realistic"
MODEL_DIR = ROOT / "models" / "it_realistic"
REPORT_DIR = ROOT / "data" / "reports"

def evaluate_configuration(config_name, train_file, test_file, class_names, normal_idx):
    print(f"\n=================================================================")
    print(f"  Evaluating Configuration: {config_name.upper()}")
    print(f"=================================================================")

    # 1. Load Data
    train_data = np.load(DATA_DIR / train_file)
    test_data = np.load(DATA_DIR / test_file)

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    print(f"Loaded {train_file}: Train shape = {X_train.shape}")
    print(f"Loaded {test_file}: Test shape = {X_test.shape}")

    # 2. Train Random Forest with identical settings to V1 baseline
    print("Training RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=2)...")
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=2,
    )
    clf.fit(X_train, y_train)

    # 3. Predict on Test Data
    print("Predicting on out-of-sample contiguous test partition...")
    y_pred = clf.predict(X_test)

    # 4. Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    class_report_str = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))

    # Explicit False-Positive Rate:
    # Normal is Negative (class_idx = normal_idx). All other 6 classes are Positive attacks.
    # TN: Actual Normal predicted as Normal
    # FP: Actual Normal predicted as any Attack class
    # FN: Actual Attack predicted as Normal
    # TP: Actual Attack predicted as any Attack class
    is_actual_normal = (y_test == normal_idx)
    is_pred_normal = (y_pred == normal_idx)

    tn = int((is_actual_normal & is_pred_normal).sum())
    fp = int((is_actual_normal & (~is_pred_normal)).sum())
    fn = int(((~is_actual_normal) & is_pred_normal).sum())
    tp = int(((~is_actual_normal) & (~is_pred_normal)).sum())

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"\n--- Results Summary for {config_name} ---")
    print(f"Accuracy:    {acc*100:.2f}% ({acc:.4f})")
    print(f"Macro F1:    {macro_f1*100:.2f}% ({macro_f1:.4f})")
    print(f"Weighted F1: {weighted_f1*100:.2f}% ({weighted_f1:.4f})")
    print(f"Benign False-Positive Rate (FPR): {fpr*100:.2f}% ({fp} / {fp + tn} normal flows falsely flagged)")
    print(f"Attack False-Negative Rate (FNR): {fnr*100:.2f}% ({fn} / {fn + tp} attacks missed as normal)")
    print("\nClassification Report:\n", class_report_str)
    print("Confusion Matrix (rows=actual, cols=predicted):\n", cm)

    # 5. Diagnostic Overlap Analysis
    train_rows = {row.tobytes() for row in X_train}
    test_rows = {row.tobytes() for row in X_test}
    exact_overlap = len(train_rows.intersection(test_rows))
    print(f"\nDiagnostic: Exact Feature-Vector Overlap across train/test = {exact_overlap} rows")

    return {
        "config_name": config_name,
        "feature_count": X_train.shape[1],
        "model": clf,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "exact_overlap": exact_overlap,
        "class_report": class_report_str,
        "confusion_matrix": cm.tolist(),
    }

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    class_names = meta["class_names"]
    normal_idx = meta["class_to_integer"]["Normal"]

    # Evaluate Configuration A (All 79 features)
    res_a = evaluate_configuration(
        config_name="Configuration A (All 79 Features)",
        train_file="it_train_all.npz",
        test_file="it_test_all.npz",
        class_names=class_names,
        normal_idx=normal_idx,
    )

    # Evaluate Configuration B (77 features - No Ports)
    res_b = evaluate_configuration(
        config_name="Configuration B (77 Features - No Ports)",
        train_file="it_train_no_ports.npz",
        test_file="it_test_no_ports.npz",
        class_names=class_names,
        normal_idx=normal_idx,
    )

    # Save trained models under dedicated realistic model directory
    joblib.dump(res_a["model"], MODEL_DIR / "random_forest_all.joblib")
    joblib.dump(res_b["model"], MODEL_DIR / "random_forest_no_ports.joblib")
    print(f"\n[OK] Models saved to: {MODEL_DIR / 'random_forest_all.joblib'} and {MODEL_DIR / 'random_forest_no_ports.joblib'}")

    # Build Comparative Final Report
    report_text = f"""RESINET-IDAM: REALISTIC IT BENCHMARK EVALUATION REPORT
======================================================================
Evaluation Date: 2026-10-23
Split Methodology: Contiguous Sequential 80/20 Block Split (Row-Order Temporal Proxy)
Model: RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=2)

PRIMARY RESEARCH LIMITATION:
This evaluation uses a row-order-based temporal-proxy split along the contiguous blocks
of IDS2025.xlsx. It prevents random point-wise interpolation within bursts, but CANNOT
be claimed as a true campaign-independent or zero-day evaluation because original timestamps,
flow IDs, and campaign metadata are completely unavailable in the source dataset.

1. BENCHMARK COMPARISON TABLE
----------------------------------------------------------------------
Metric                    Frozen Baseline*    Configuration A         Configuration B
                          (Random Stratified) (Contiguous, 79 Feats)  (Contiguous, No Ports)
---------------------------------------------------------------------------------------------
Number of Features:       79                  79                      77
Split Protocol:           Random Stratified   Contiguous Sequential   Contiguous Sequential
Overall Test Accuracy:    99.56%              {res_a['accuracy']*100:6.2f}%                  {res_b['accuracy']*100:6.2f}%
Macro F1-Score:           96.76%              {res_a['macro_f1']*100:6.2f}%                  {res_b['macro_f1']*100:6.2f}%
Weighted F1-Score:        99.56%              {res_a['weighted_f1']*100:6.2f}%                  {res_b['weighted_f1']*100:6.2f}%
Benign FPR (Normal=Neg):  0.15%               {res_a['fpr']*100:6.2f}%                   {res_b['fpr']*100:6.2f}%
Attack FNR (Missed=Norm): 0.44%               {res_a['fnr']*100:6.2f}%                   {res_b['fnr']*100:6.2f}%
Exact Vector Overlap:     0 rows              {res_a['exact_overlap']} row(s)                  {res_b['exact_overlap']} rows

*Note on Frozen Baseline: The random-split baseline is included strictly as a reference point.
It is NOT a directly equivalent test because random stratified splitting interpolates flows
within continuous attack bursts, whereas the contiguous split forces out-of-burst extrapolation.

2. DIAGNOSTIC FEATURE-VECTOR OVERLAP ANALYSIS
----------------------------------------------------------------------
- Configuration A (All 79 Features): Exactly {res_a['exact_overlap']} overlapping row across train and test.
- Configuration B (No Ports - 77 Features): Exactly {res_b['exact_overlap']} overlapping rows across train and test.

Interpretation:
The 824 exact overlaps in Configuration B are a critical diagnostic finding. When ephemeral
port numbers (which change dynamically with every new TCP connection) are excluded, 824 flow
records in the test set have identical feature vectors to records in the training set (identical
packet counts, byte volumes, header lengths, and TCP flag profiles).
Because original flow IDs and timestamps are unavailable, we cannot definitively prove whether
these 824 rows represent identical duplicate observations from the same capture script or
legitimate repeated protocol behavior (e.g., standard DNS queries or identical keepalives).
They are preserved as-is without deduplication to maintain an unbiased out-of-burst evaluation.

3. CONFIGURATION A DETAILED REPORT (ALL 79 FEATURES)
----------------------------------------------------------------------
FPR Definition: Normal = Negative (5,237 flows), Attacks = Positive (13,107 flows)
- True Negatives (TN): {res_a['tn']}
- False Positives (FP): {res_a['fp']}  --> FPR = {res_a['fpr']*100:.4f}%
- False Negatives (FN): {res_a['fn']}  --> FNR = {res_a['fnr']*100:.4f}%
- True Positives (TP): {res_a['tp']}

Classification Report:
{res_a['class_report']}

Confusion Matrix (rows = true label, cols = predicted label):
Classes: {class_names}
{np.array(res_a['confusion_matrix'])}

4. CONFIGURATION B DETAILED REPORT (NO PORTS - 77 FEATURES)
----------------------------------------------------------------------
FPR Definition: Normal = Negative (5,237 flows), Attacks = Positive (13,107 flows)
- True Negatives (TN): {res_b['tn']}
- False Positives (FP): {res_b['fp']}  --> FPR = {res_b['fpr']*100:.4f}%
- False Negatives (FN): {res_b['fn']}  --> FNR = {res_b['fnr']*100:.4f}%
- True Positives (TP): {res_b['tp']}

Classification Report:
{res_b['class_report']}

Confusion Matrix (rows = true label, cols = predicted label):
Classes: {class_names}
{np.array(res_b['confusion_matrix'])}

5. SCIENTIFIC CONCLUSIONS
----------------------------------------------------------------------
1. Effect of Out-of-Burst Splitting: Evaluating on the later 20% of contiguous label blocks
   eliminates the point-wise interpolation artifact inherent in random stratified splits.
2. Effect of Port Sanitization: Removing Source and Destination Ports evaluates whether the
   model relies on port-memorization shortcuts (which yielded 88.15% decision tree accuracy)
   or true behavioral traffic dynamics.
3. Frozen V1.0 Integrity: All baseline V1.0 reports, datasets, and models remain preserved
   and unchanged in their respective directories.
"""

    report_path = REPORT_DIR / "it_realistic_evaluation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[OK] Full realistic evaluation report written to: {report_path}")

if __name__ == "__main__":
    main()
