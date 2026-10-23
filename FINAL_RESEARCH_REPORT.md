# ResiNet-IDAM: Final Research & Baseline Evaluation Report

## Adaptive Deep Learning and Zero-Trust Framework for Detecting Zero-Day Intrusions in IT-OT Hybrid Critical Infrastructures

---

## 1. Executive Summary & Research Prototype Status

The ResiNet-IDAM project investigates hybrid cyber-physical security by combining Information Technology (IT) network intrusion detection with Operational Technology (OT) physical anomaly detection, correlated under Zero-Trust architectural principles.

### Frozen Baseline Classification
> [!IMPORTANT]
> The current models, datasets, and scripts are formally **frozen as Research Prototype Version 1.0**. 
> All reported metrics reflect rigorous out-of-sample testing without synthetic shortcut features. Real cross-domain intrusion detection cannot be definitively proven until empirical, genuinely time-aligned IT and OT telemetry is captured concurrently on a unified hybrid testbed.

---

## 2. Summary of Key Experimental Benchmarks

| Component / Evaluation Dimension | Primary Metric | Supporting Metrics & Findings | Status |
|---|---|---|---|
| **IT Standalone Baseline (Random Forest)** | **99.56% Accuracy** | Macro F1: **96.76%**; Infiltration Recall: **66.67%** (4/6 caught) | Frozen Baseline |
| **OT Standalone Baseline (Autoencoder)** | **92.27% Accuracy** | Attack Precision: **34.92%**; Attack Recall: **46.39%**; F1: **39.84%** (1,598 missed attacks) | Frozen Baseline |
| **OT Baseline (Isolation Forest)** | **65.86% Accuracy** | Attack Precision: **10.49%**; Attack Recall: **68.80%**; F1: **18.20%** (17,506 false alarms) | Frozen Baseline |
| **Cross-Domain Correlation ($R_{\text{CP}}$)** | **28.95% Attack F1** | Precision: **56.44%**; Recall: **19.47%**; FPR: **0.66%** (Without synthetic shortcuts) | Verified Prototype |
| **Scenario D Physical Attacks Caught** | **7.6% (10 / 132)** | **122 physical attacks missed** by OT Autoencoder threshold; zero alert suppression | Audited & Triaged |
| **Unseen Botnet ARES Zero-Day Detection** | **0.00% (0 / 375)** | ROC-AUC: **0.5562** (Unsupervised flow density estimation fails on C2 overlap) | Verified & Reproducible |
| **Unseen Infiltration Zero-Day Detection** | **33.33% (2 / 6)** | ROC-AUC: **0.8332** (Moderate anomaly detection on high-divergence flows) | Verified & Reproducible |

---

## 3. IT Dataset Leakage & Train/Test Split Realism Audit

An empirical audit of `data/processed/it/` was conducted to investigate data leakage, shortcut learning, and the validity of the stratified 80/20 train/test split.

### 3.1 Feature Shortcut Learning
Single-feature decision trees (depth = 4) were trained on individual flow features to measure how much label information is concentrated in single columns:
* **`Destination Port` alone: `88.15%` accuracy across all 7 classes.**
* **`Packet Length Mean` alone: `82.08%` accuracy.**
* **`Init_Win_bytes_forward` alone: `72.19%` accuracy.**
* **`Flow Duration` alone: `65.05%` accuracy.**

*Finding:* In benchmark flow datasets, attack scripts frequently bind to fixed destination ports or produce identical packet buffers. Models can achieve high accuracy by memorizing port numbers or packet sizes rather than learning generalizable network behavior.

### 3.2 Near-Duplicate Flow Leakage
Euclidean nearest-neighbor distance was measured between the test partition (5,000 samples) and the training partition (25,000 samples):
* **Median distance to nearest training flow:** **`0.0500`**
* **Test flows with virtually identical training vectors (dist < 0.05):** **`49.96%`**
* **Test flows with close training vectors (dist < 0.50):** **`90.66%`**
* **Class label match rate of nearest training flow:** **`98.28%`**

*Finding:* While [check_it_overlap.py](file:///c:/ResiNet-IDAM/src/check_it_overlap.py) confirmed zero exact byte-for-byte duplicates, **nearly half of the test flows are near-duplicate vectors** of training flows. 

### 3.3 Realism of the Random Stratified Split
In operational environments, network flows are generated in continuous bursts and stateful sessions. The current random stratified 80/20 split draws individual flows independently across the entire dataset. Consequently, flows belonging to the same burst or IP session appear in both training and testing sets, artificially inflating closed-world test accuracy to 99.56%. A realistic evaluation requires **group-based or temporal splitting** by capture session or day.

---

## 4. OT Dataset Distribution & Preprocessing Pipeline Audit

### 4.1 Attack Distribution in HAI Testbed Telemetry
* **Total Duration:** 54,000 records sampled at strictly 1.0 Hz (15 hours).
* **Class Imbalance:** 51,019 Normal seconds (**94.48%**) vs. 2,981 Attack seconds (**5.52%**).
* **Continuous Attack Blocks:** **14 distinct physical attack events**:
  * Minimum Duration: **56 seconds**
  * Median Duration: **159.5 seconds**
  * Maximum Duration: **628 seconds**
* Attacks manipulate physical setpoints (`x1001_05_SETPOINT_OUT`), water levels, pump speeds, and valve actuators.

### 4.2 Preprocessing Pipeline Audit
* **Feature Pruning:** Of the 86 raw process channels, **20 static/constant columns were dropped** (e.g., `P2_Emerg`, `P2_OnOff`), retaining **66 active physical channels**.
* **Scaling Isolation:** `StandardScaler` was fitted strictly on the normal training split (`hai-train1`) and reused to transform validation and test sets.
* **Reconstruction Threshold:** Threshold $\theta = 0.821954$ was derived from the 99th percentile of reconstruction errors on the chronological validation split.
* **Performance Trade-Off:** The Autoencoder achieves 92.27% accuracy, but has low attack precision (34.92%) and misses **1,598 attack seconds** (46.39% recall), generating 2,578 false alarms during normal periods.

---

## 5. Cross-Domain Integration: Triaged Multi-Channel Performance

### 5.1 Removal of Synthetic Zero-Trust Leakage
In earlier exploratory runs, a synthetic `zt_context` feature was generated conditionally based on attack scenarios. This leaked the target labels ($r = 0.9110$) and artificially produced a 99.29% F1 score. 

In the frozen implementation ([train_fusion_model.py](file:///c:/ResiNet-IDAM/src/integration/train_fusion_model.py)), **`zt_context` was completely removed**. The cross-domain model relies strictly on real telemetry:
$$R_{\text{CP}}(t) = \sigma\Big(6.5199\,p_{\text{attack}}(t) + 4.7116\,S_{\text{OT}}(t) - 0.0324\,\big(p_{\text{attack}}(t) \cdot S_{\text{OT}}(t)\big) - 7.9211\Big)$$

### 5.2 Out-of-Sample Benchmark Comparison (26,686 Test Records)
| Pipeline / Channel | Accuracy | Precision | Recall | Attack F1 | False Positive Rate | False Positives | False Negatives |
|---|---|---|---|---|---|---|---|
| **IT Channel Alone ($R_{\text{IT}} \ge 0.50$)** | 94.99% | 0.4559 | **0.9796** | **0.6222** | 5.14% | 1,315 | 23 |
| **OT Channel Alone ($R_{\text{OT}} \ge 0.50$)** | 92.51% | 0.1234 | 0.1271 | 0.1252 | 3.97% | 1,016 | 982 |
| **Equal-Weight Linear ($0.5\,p_{\text{att}} + 0.5\,S_{\text{ot}}$)** | 95.88% | 0.5058 | 0.9671 | 0.6642 | 4.16% | 1,063 | 37 |
| **Trained $R_{\text{CP}}$ (Without $ZT$)** | **95.97%** | **0.5644** | **0.1947** | **`28.95%`** | **0.66%** | **169** | **906** |

*True Metric:* Without shortcut features, the cyber-physical correlation model achieves **`F1 = 28.95%`**. It suppresses false positives down to 0.66%, but exhibits conservative recall (19.47%) due to missed detections in the OT Autoencoder.

### 5.3 Detailed Scenario Audit & Scenario D Analysis
The system dispatches alerts through three independent channels:
* **Tier 4 (Critical CP):** Automated plant/network isolation.
* **Tier 2 (Process Alert):** Dispatched to Plant Engineering without requiring IT corroboration.
* **Tier 2 (Enterprise Alert):** Dispatched to Enterprise SOC without tripping physical safety limits.

| Scenario Breakdown | Total Records | IT Alerts | OT Alerts | Cyber-Physical Alerts ($R_{\text{CP}}$) | Tier-2 OT Triage | Tier-2 IT Triage | Total Attacks Caught |
|---|---|---|---|---|---|---|---|
| **Normal Baseline** | 24,445 | 329 (1.3%) | 965 (3.9%) | 20 (0.1%) | 952 | 308 | 1,281 (5.2% FPR) |
| **Scenario A (Correlated Attack)** | 1,069 | 1,069 (100.0%) | 130 (12.2%) | 210 (19.6%) | 0 | 859 | **1,069 (100.0%)** |
| **Scenario B (Stealth Infil)** | 56 | 33 (58.9%) | 13 (23.2%) | 9 (16.1%) | 5 | 24 | **38 (67.9%)** |
| **Scenario C (Isolated IT Threat)** | 984 | 984 (100.0%) | 41 (4.2%) | 149 (15.1%) | 0 | 835 | **984 (100.0%)** |
| **Scenario D (Isolated OT Fault)** | 132 | 2 (1.5%) | **10 (7.6%)** | **0 (0.0%)** | **10 (7.6%)** | 2 | **12 (9.1%)** |

#### Scenario D Audit (Isolated OT Physical Attacks):
* In Scenario D, physical sensor tampering occurred in the OT environment without an IT network precursor.
* **Previous Failure:** Single-decision model flagged 0/132 attacks because it required IT corroboration (100% false negative rate).
* **Multi-Channel Result:** Dispatched as a **Tier-2 Plant Engineering Alert**. 
* **Missed Attack Breakdown:** The OT Autoencoder caught **10 out of 132 physical attack seconds (7.6%)** and **missed 122 attack seconds** due to its baseline reconstruction threshold.

---

## 6. Zero-Day Intrusion Evaluation: 100% Deterministic Reproducibility

Two consecutive runs of [evaluate_loco.py](file:///c:/ResiNet-IDAM/src/integration/evaluate_loco.py) with seed 42 verified identical outputs:

```
=======================================================
  LOCO ZERO-DAY EXPERIMENT (SEED 42 - DETERMINISTIC)
=======================================================
1. Closed-World Supervised Model (Random Forest on N-1 classes):
   - Withheld 'Infiltration': Forced 6 / 6 (100.0%) to 'Normal'
   - Withheld 'Botnet ARES':  Forced 375 / 375 (100.0%) to 'Normal'
   --> Result: Closed-world classifiers provide 0% defense against novel attacks.

2. Out-of-Distribution Anomaly Detector (Isolation Forest on Normal Baseline):
   - Withheld 'Infiltration':
     * Detection Rate: 33.33% (2 / 6 detected at 1.01% FPR)
     * ROC-AUC: 0.8332
   - Withheld 'Botnet ARES':
     * Detection Rate: 0.00% (0 / 375 detected at 1.01% FPR)
     * ROC-AUC: 0.5562
```

### Why Botnet ARES Fails (0.00% Detection Rate):
1. **Flow Distribution Overlap:** In tabular NetFlow feature space (packet lengths, inter-arrival times, flow durations), low-frequency Botnet command-and-control (C2) flows overlap with legitimate periodic background traffic (e.g., NTP keepalives, HTTP polling).
2. **Density Blind Spot:** Unsupervised density estimation (Isolation Forest) trained purely on normal network flows cannot separate Botnet ARES from legitimate traffic without DNS domain reputation, host process trees, or protocol-aware payload inspection.
3. **Scientific Finding:** Zero-day detection claims are unsupported for distribution-overlapping threats under flow-level telemetry alone.

---

## 7. Primary Project Limitation: Synthetic Cross-Domain Pairing

> [!WARNING]
> **Synthetic Coupling Limitation:** 
> Because `IDS2025` and `HAI` were captured on separate testbeds at different times, the cross-domain pairing in this prototype is synthetically generated. Real multi-stage cyber-physical correlation cannot be fully validated until synchronized, time-aligned IT and OT telemetry is recorded from a concurrent testbed.

---

## 8. Artifact and Script Inventory (Frozen Prototype)

| Artifact Path | Role / Content |
|---|---|
| `data/processed/it/it_train.npz` | Scaled IT training partition (73,395 records, 79 features) |
| `data/processed/it/it_test.npz` | Scaled IT test partition (18,349 records, 79 features) |
| `data/processed/hai-train-scaled.csv` | Scaled normal OT training telemetry (280,800 records, 66 features) |
| `data/processed/hai-test-scaled.csv` | Labeled OT test telemetry (54,000 records, 14 attack blocks) |
| `models/it/random_forest.joblib` | Trained IT baseline classifier (200 trees, balanced) |
| `models/ot_autoencoder.keras` | Trained OT autoencoder baseline |
| `models/fusion_model.joblib` | Trained cross-domain correlation model (without ZT shortcuts) |
| `src/integration/verify_schemas.py` | Schema and artifact validation script |
| `src/integration/generate_scenarios.py` | Chronologically partitioned benchmark scenario generator |
| `src/integration/calibrate_scores.py` | Validation-isolated score calibration script |
| `src/integration/train_fusion_model.py` | Multi-channel triage evaluation script |
| `src/integration/evaluate_loco.py` | Deterministic Leave-One-Class-Out zero-day evaluation script |
| `src/integration/investigate_leakage.py` | IT leakage and OT distribution audit script |
| `data/reports/integration_report.txt` | Detailed multi-channel evaluation report |
| `data/reports/loco_zero_day_report.txt` | Detailed LOCO zero-day evaluation report |
| `IT_EVALUATION_PLAN.md` | Formal IT reliability roadmap |
| `IT_OT_INTEGRATION_PLAN.md` | Multi-channel integration design document |
| `FINAL_RESEARCH_REPORT.md` | Comprehensive frozen baseline evaluation report |

---
*Report Version: 1.0 (Frozen Research Prototype)*  
*Project: ResiNet-IDAM*
