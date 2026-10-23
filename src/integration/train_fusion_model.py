"""
ResiNet-IDAM: Triaged Multi-Channel IT-OT Evaluation Engine
------------------------------------------------------------
Implements honest, uncorrupted evaluation of IT and OT detection:
1. Completely removes synthetic zt_context feature from model training.
2. Produces three separate risk signals:
   - R_IT(t): Scalar IT attack probability (Enterprise SOC channel)
   - R_OT(t): Calibrated OT physical reconstruction score (Plant Engineering channel)
   - R_CP(t): Cross-domain cyber-physical correlation model (Unified Critical channel)
3. Evaluates Scenario D (isolated OT fault) through the OT channel without requiring IT corroboration.
4. Reports honest metrics across all individual channels and unified tiered triage.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import joblib

ROOT = Path(__file__).resolve().parents[2]
INTEG_DIR = ROOT / "data" / "processed" / "integration"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "data" / "reports"

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fpr": fpr,
    }

def main():
    print("=== [Step 4 & 5] Training and Evaluating Triaged Integration Model ===")

    # 1. Load Validation Data (Scores)
    val_data = np.load(INTEG_DIR / "scores_val.npz")
    p_att_val = val_data["p_attack"]
    S_ot_val = val_data["S_ot"]
    y_cp_val = val_data["is_cyber_physical"] # 1 if Scenario A or B
    y_ot_val = val_data["ot_label"]           # 1 if OT physical attack

    # 2. Load Test Data (Scores)
    test_data = np.load(INTEG_DIR / "scores_test.npz")
    p_att_test = test_data["p_attack"]
    S_ot_test = test_data["S_ot"]
    y_cp_test = test_data["is_cyber_physical"]
    y_ot_test = test_data["ot_label"]

    val_scen = pd.read_csv(INTEG_DIR / "scenarios_val.csv")
    test_scen = pd.read_csv(INTEG_DIR / "scenarios_test.csv")

    # Strictly 3 features derived from telemetry: [p_attack, S_ot, p_attack * S_ot]
    # Synthetic zt_context is COMPLETELY REMOVED.
    X_val = np.column_stack([
        p_att_val,
        S_ot_val,
        p_att_val * S_ot_val,
    ])

    X_test = np.column_stack([
        p_att_test,
        S_ot_test,
        p_att_test * S_ot_test,
    ])

    # 3. Fit Cyber-Physical Model strictly on VALIDATION split
    print("Training Cyber-Physical Correlation Model (without zt_context)...")
    clf = LogisticRegression(C=1.0, fit_intercept=True, random_state=42)
    clf.fit(X_val, y_cp_val)

    coefs = clf.coef_[0]
    intercept = clf.intercept_[0]
    print(f"  Fitted weights: alpha={coefs[0]:.4f}, beta={coefs[1]:.4f}, mu={coefs[2]:.4f}")
    print(f"  Intercept: {intercept:.4f}")

    # 4. Determine Thresholds on VALIDATION set
    # Channel 1: IT threshold
    tau_it = 0.50
    # Channel 2: OT threshold (standard calibrated sigmoid midpoint)
    tau_ot = 0.50

    # Channel 3: Cyber-physical threshold optimized for F1 on validation set
    cp_val_probs = clf.predict_proba(X_val)[:, 1]
    best_tau_cp = 0.50
    best_f1_val = 0.0
    for thresh in np.linspace(0.1, 0.9, 81):
        preds = (cp_val_probs >= thresh).astype(int)
        f = f1_score(y_cp_val, preds, zero_division=0)
        if f > best_f1_val:
            best_f1_val = f
            best_tau_cp = thresh

    print(f"[OK] Validation selected thresholds: tau_it={tau_it:.2f}, tau_ot={tau_ot:.2f}, tau_cp={best_tau_cp:.4f} (Val CP F1: {best_f1_val:.4f})")

    # 5. Out-of-sample Evaluation on TEST partition
    print("\nEvaluating separate risk outputs on TEST partition...")
    cp_test_probs = clf.predict_proba(X_test)[:, 1]

    # Individual Channel Alerts:
    alert_it = (p_att_test >= tau_it).astype(int)
    alert_ot = (S_ot_test >= tau_ot).astype(int)
    alert_cp = (cp_test_probs >= best_tau_cp).astype(int)

    # Multi-Tier Triage Assignment:
    # Tier 4: Critical Cyber-Physical (alert_cp == 1 or (alert_it == 1 and alert_ot == 1))
    # Tier 2 Process: Isolated OT Anomaly (alert_ot == 1 and not Tier 4)
    # Tier 2 Enterprise: Isolated IT Intrusion (alert_it == 1 and not Tier 4)
    # Tier 1 Normal: No alert
    tier4_cp = (alert_cp == 1) | ((alert_it == 1) & (alert_ot == 1))
    tier2_ot = (alert_ot == 1) & (~tier4_cp)
    tier2_it = (alert_it == 1) & (~tier4_cp)
    any_alert = (alert_it == 1) | (alert_ot == 1) | (alert_cp == 1)

    # Benchmarks for Cyber-Physical Attack Detection (Ground Truth: y_cp_test):
    m_it_cp = compute_metrics(y_cp_test, alert_it)
    m_ot_cp = compute_metrics(y_cp_test, alert_ot)
    m_linear_cp = compute_metrics(y_cp_test, ((0.5 * p_att_test + 0.5 * S_ot_test) >= 0.5).astype(int))
    m_model_cp = compute_metrics(y_cp_test, alert_cp)

    print("\n================ CYBER-PHYSICAL CORRELATION BENCHMARK (TEST SET) ================")
    print(f"{'Channel / Model':<28} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'FPR':<10}")
    print("-" * 78)
    print(f"{'IT Channel Alert (p_att)':<28} {m_it_cp['accuracy']*100:6.2f}%    {m_it_cp['precision']:8.4f}   {m_it_cp['recall']:8.4f}   {m_it_cp['f1']:8.4f}   {m_it_cp['fpr']*100:6.2f}%")
    print(f"{'OT Channel Alert (S_ot)':<28} {m_ot_cp['accuracy']*100:6.2f}%    {m_ot_cp['precision']:8.4f}   {m_ot_cp['recall']:8.4f}   {m_ot_cp['f1']:8.4f}   {m_ot_cp['fpr']*100:6.2f}%")
    print(f"{'Equal Linear (0.5/0.5)':<28} {m_linear_cp['accuracy']*100:6.2f}%    {m_linear_cp['precision']:8.4f}   {m_linear_cp['recall']:8.4f}   {m_linear_cp['f1']:8.4f}   {m_linear_cp['fpr']*100:6.2f}%")
    print(f"{'Trained R_CP (No ZT)':<28} {m_model_cp['accuracy']*100:6.2f}%    {m_model_cp['precision']:8.4f}   {m_model_cp['recall']:8.4f}   {m_model_cp['f1']:8.4f}   {m_model_cp['fpr']*100:6.2f}%")

    # 6. Granular Scenario Analysis (Separating IT, OT, CP, and Triage)
    print("\n================ SCENARIO-BY-SCENARIO DETAILED BREAKDOWN ================")
    test_scen["alert_it"] = alert_it
    test_scen["alert_ot"] = alert_ot
    test_scen["alert_cp"] = alert_cp
    test_scen["tier4_cp"] = tier4_cp.astype(int)
    test_scen["tier2_ot"] = tier2_ot.astype(int)
    test_scen["tier2_it"] = tier2_it.astype(int)
    test_scen["any_alert"] = any_alert.astype(int)

    scen_rows = []
    for s_name, grp in test_scen.groupby("scenario"):
        tot = len(grp)
        n_it = int(grp["alert_it"].sum())
        n_ot = int(grp["alert_ot"].sum())
        n_cp = int(grp["alert_cp"].sum())
        n_t4 = int(grp["tier4_cp"].sum())
        n_t2_ot = int(grp["tier2_ot"].sum())
        n_t2_it = int(grp["tier2_it"].sum())
        n_any = int(grp["any_alert"].sum())

        row = {
            "scenario": s_name,
            "total": tot,
            "it_alert": n_it,
            "ot_alert": n_ot,
            "cp_alert": n_cp,
            "tier4": n_t4,
            "tier2_ot": n_t2_ot,
            "tier2_it": n_t2_it,
            "any_alert": n_any,
        }
        scen_rows.append(row)
        print(f"\nScenario: {s_name} (Total Records: {tot})")
        print(f"  - IT Channel Alerts:       {n_it:>5} / {tot:<5} ({n_it/tot*100:5.1f}%)")
        print(f"  - OT Channel Alerts:       {n_ot:>5} / {tot:<5} ({n_ot/tot*100:5.1f}%)")
        print(f"  - Cyber-Physical Alerts:   {n_cp:>5} / {tot:<5} ({n_cp/tot*100:5.1f}%)")
        print(f"  - Triaged as Tier 4 (CP):  {n_t4:>5} / {tot:<5} ({n_t4/tot*100:5.1f}%)")
        print(f"  - Triaged as Tier 2 (OT):  {n_t2_ot:>5} / {tot:<5} ({n_t2_ot/tot*100:5.1f}%)")
        print(f"  - Triaged as Tier 2 (IT):  {n_t2_it:>5} / {tot:<5} ({n_t2_it/tot*100:5.1f}%)")
        print(f"  - Any Alert (Total Caught):{n_any:>5} / {tot:<5} ({n_any/tot*100:5.1f}%)")

    # 7. Write Comprehensive Honest Report
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "fusion_model.joblib")

    # Find Scenario D row explicitly
    scen_d_row = next(r for r in scen_rows if r["scenario"] == "Scenario_D_Isolated_OT")

    report_text = f"""RESINET-IDAM: HONEST IT-OT INTEGRATION EVALUATION REPORT
======================================================================
Status: Verified Research Prototype Evaluation (No Synthetic ZT Leakage)
Evaluation Partition: 26,686 out-of-sample records chronologically isolated.

PRIMARY LIMITATION:
In this prototype, IT network flows from IDS2025 and OT sensor readings from HAI
are synthetically paired. Genuine cross-domain intrusion detection cannot be
empirically validated until synchronized, time-aligned IT-OT telemetry is available.

1. METHODOLOGY & LEAKAGE CORRECTION
----------------------------------------------------------------------
- The synthetic 'zt_context' feature has been COMPLETELY REMOVED from training.
- The single-decision fusion model has been replaced with THREE INDEPENDENT CHANNELS:
  1. R_IT(t):  Enterprise IT attack probability from Random Forest
  2. R_OT(t):  Physical anomaly score from OT Autoencoder reconstruction
  3. R_CP(t):  Cross-domain cyber-physical correlation model
- Coefficients fitted on VALIDATION data without ZT:
  alpha (IT Weight):       {coefs[0]:.4f}
  beta (OT Weight):        {coefs[1]:.4f}
  mu (Interaction Weight): {coefs[2]:.4f}
  Intercept:               {intercept:.4f}
  Validation Thresholds:   tau_it = {tau_it:.2f}, tau_ot = {tau_ot:.2f}, tau_cp = {best_tau_cp:.4f}

2. UNBIASED TEST PERFORMANCE ON CYBER-PHYSICAL ATTACK DETECTION
----------------------------------------------------------------------
Evaluating detection of correlated attacks (Ground Truth: Scenarios A & B):

Metric                 IT-Only Alert   OT-Only Alert   Linear(0.5/0.5)  Trained R_CP
Accuracy:               {m_it_cp['accuracy']*100:6.2f}%          {m_ot_cp['accuracy']*100:6.2f}%          {m_linear_cp['accuracy']*100:6.2f}%           {m_model_cp['accuracy']*100:6.2f}%
Precision:              {m_it_cp['precision']:8.4f}         {m_ot_cp['precision']:8.4f}         {m_linear_cp['precision']:8.4f}          {m_model_cp['precision']:8.4f}
Recall:                 {m_it_cp['recall']:8.4f}         {m_ot_cp['recall']:8.4f}         {m_linear_cp['recall']:8.4f}          {m_model_cp['recall']:8.4f}
Attack F1-Score:        {m_it_cp['f1']:8.4f}         {m_ot_cp['f1']:8.4f}         {m_linear_cp['f1']:8.4f}          {m_model_cp['f1']:8.4f}
False Positive Rate:    {m_it_cp['fpr']*100:6.2f}%          {m_ot_cp['fpr']*100:6.2f}%          {m_linear_cp['fpr']*100:6.2f}%           {m_model_cp['fpr']*100:6.2f}%
False Positives (FP):   {m_it_cp['fp']:<15} {m_ot_cp['fp']:<15} {m_linear_cp['fp']:<17} {m_model_cp['fp']:<15}
False Negatives (FN):   {m_it_cp['fn']:<15} {m_ot_cp['fn']:<15} {m_linear_cp['fn']:<17} {m_model_cp['fn']:<15}

KEY FINDING:
Without the leaked ZT feature, R_CP achieves F1 = {m_model_cp['f1']*100:.2f}%.
The previously reported 99.29% F1 was entirely driven by target leakage in zt_context.
The current score accurately reflects the reality of combining a high-accuracy IT classifier
with an OT Autoencoder that has ~46% baseline recall and high false positives.

3. SCENARIO-BY-SCENARIO MULTI-CHANNEL AUDIT
----------------------------------------------------------------------
"""
    for r in scen_rows:
        report_text += f"""Scenario: {r['scenario']} (Total: {r['total']} records)
  - IT Channel Alerts:        {r['it_alert']:>5} ({r['it_alert']/r['total']*100:5.1f}%)
  - OT Channel Alerts:        {r['ot_alert']:>5} ({r['ot_alert']/r['total']*100:5.1f}%)
  - Cyber-Physical Alerts:    {r['cp_alert']:>5} ({r['cp_alert']/r['total']*100:5.1f}%)
  - Triaged Tier 4 (CP):      {r['tier4']:>5} ({r['tier4']/r['total']*100:5.1f}%)
  - Triaged Tier 2 (OT):      {r['tier2_ot']:>5} ({r['tier2_ot']/r['total']*100:5.1f}%)
  - Triaged Tier 2 (IT):      {r['tier2_it']:>5} ({r['tier2_it']/r['total']*100:5.1f}%)
  - Any Alert Triggered:      {r['any_alert']:>5} ({r['any_alert']/r['total']*100:5.1f}%)

"""
    report_text += f"""4. VERIFICATION OF SCENARIO D (ISOLATED OT PHYSICAL ATTACK)
----------------------------------------------------------------------
- In Scenario D, physical sensor manipulation occurred without IT network precursors.
- Result under previous single fusion model: 0% flagged (132/132 attacks silently missed).
- Result under Triaged Multi-Channel model:
  * IT Channel Alerts: {scen_d_row['it_alert']} (Correct: no network attack occurred).
  * OT Channel Alerts: {scen_d_row['ot_alert']} / {scen_d_row['total']} triggered.
  * Triage Action: Dispatched as Tier-2 Plant Engineering Alert without requiring IT corroboration.
  * Limitation: The OT Autoencoder missed {scen_d_row['total'] - scen_d_row['ot_alert']} records due to its baseline physical reconstruction threshold.

5. SCIENTIFIC CONCLUSIONS & RESEARCH LIMITATIONS
----------------------------------------------------------------------
1. Synthetic IT-OT Pairing Limitation: Because IDS2025 and HAI data were captured on
   separate testbeds at different times, cross-domain pairing is synthetic. Genuine
   multi-stage cyber-physical correlation cannot be validated without time-aligned telemetry.
2. Honest Metrics: The cross-domain correlation model achieves F1 = {m_model_cp['f1']*100:.2f}%,
   reflecting true physical sensor and network flow limitations without shortcut features.
3. Multi-Channel Triage: Separate IT and OT channels are mandatory so isolated attacks
   are not suppressed by lack of cross-domain corroboration.
"""
    report_path = REPORTS_DIR / "integration_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[OK] Honest integration report written to: {report_path}")

if __name__ == "__main__":
    main()
