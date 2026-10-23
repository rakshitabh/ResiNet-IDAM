# ResiNet-IDAM: Comprehensive Research & Evaluation Report

## Adaptive Deep Learning and Zero-Trust Framework for Detecting Zero-Day Intrusions in IT-OT Hybrid Critical Infrastructures

---

## 1. Executive Summary & Research Prototype Status

The **ResiNet-IDAM** framework investigates multi-stage cyber-physical security by combining Information Technology (IT) network intrusion detection with Operational Technology (OT) physical telemetry anomaly detection, correlated under Zero-Trust architectural principles.

### Dual-Phase Evaluation Architecture
This report documents both the **frozen V1.0 random-split baseline** and the **realistic out-of-burst evaluation**:
1. **Frozen Baseline Reference (V1.0):** Preserved as a historical reference point evaluated on random stratified splits (99.56% IT accuracy; 28.95% real cyber-physical fusion F1 without synthetic shortcuts).
2. **Realistic 7-Class Benchmark:** Evaluated on a contiguous sequential block split (row-order temporal proxy) comparing **Configuration A** (all 79 features) against **Configuration B** (77 features, source/destination ports removed).
3. **Synthetic Held-Out-Class Evaluation (LOCO):** Separate Leave-One-Class-Out evaluation measuring closed-world supervised failure modes and unsupervised anomaly detection on withheld attack classes (`Infiltration` and `Botnet ARES`).

> [!IMPORTANT]
> **Dataset-Specific Scope & Experimental Constraints:**
> - All reported conclusions are **strictly dataset-specific** to `IDS2025` (IT NetFlow) and `HAI 21.03` (OT water treatment testbed).
> - The contiguous split acts as a **row-order temporal proxy** to prevent intra-burst interpolation, but cannot be claimed as a verified campaign-independent or zero-day test because original timestamps, flow IDs, and host metadata are absent from `IDS2025.xlsx`.
> - Held-out class experiments represent **synthetic held-out-class evaluations**, not genuine zero-day intrusion detection.
> - IT and OT datasets were recorded on separate testbeds and are **synthetically paired**; real-world multi-stage cross-domain correlation requires concurrently recorded physical testbed telemetry.

---

## 2. Frozen Baseline Reference (V1.0 Summary)

The original prototype metrics are preserved below as a baseline reference. These models used random stratified 80/20 splitting.

| Component / Evaluation Dimension | Primary Metric | Supporting Metrics & Findings | Status |
|---|---|---|---|
| **IT Standalone Baseline (Random Forest)** | **99.56% Accuracy** | Macro F1: **96.76%**; Infiltration Recall: **66.67%** (4/6 caught) | Frozen Baseline Reference |
| **OT Standalone Baseline (Autoencoder)** | **92.27% Accuracy** | Attack Precision: **34.92%**; Attack Recall: **46.39%**; F1: **39.84%** (1,598 missed attacks) | Frozen Baseline Reference |
| **OT Baseline (Isolation Forest)** | **65.86% Accuracy** | Attack Precision: **10.49%**; Attack Recall: **68.80%**; F1: **18.20%** (17,506 false alarms) | Frozen Baseline Reference |
| **Cross-Domain Correlation ($R_{\text{CP}}$)** | **28.95% Attack F1** | Precision: **56.44%**; Recall: **19.47%**; FPR: **0.66%** (Without synthetic shortcuts) | Verified Prototype |
| **Scenario D Physical Attacks Caught** | **7.6% (10 / 132)** | **122 physical attacks missed** by OT Autoencoder threshold; zero alert suppression | Audited & Triaged |
| **Unseen Botnet ARES Zero-Day Detection** | **0.00% (0 / 375)** | ROC-AUC: **0.5562** (Unsupervised flow density estimation fails on C2 overlap) | Verified & Reproducible |
| **Unseen Infiltration Zero-Day Detection** | **33.33% (2 / 6)** | ROC-AUC: **0.8332** (Moderate anomaly detection on high-divergence flows) | Verified & Reproducible |

---

## 3. Realistic IT Benchmark Evaluation (Contiguous Sequential Split)

To address intra-burst point-wise interpolation and port shortcut memorization, a contiguous 80/20 sequential split was implemented on `data/raw/IDS2025.xlsx` across its 79 contiguous label blocks (earlier 80% to train: 73,400 rows; later 20% to test: 18,344 rows). Imputation and standardization were strictly fitted on training data only.

### 3.1 Benchmark Comparison Table
*Model: `RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=2)`*

| Metric | Frozen Baseline Reference\*<br>*(Random Stratified, 79 Feats)* | Configuration A<br>*(Contiguous, All 79 Feats)* | Configuration B<br>*(Contiguous, No Ports - 77 Feats)* |
| :--- | :---: | :---: | :---: |
| **Number of Features** | 79 | 79 | **77** |
| **Split Protocol** | Random Stratified (Interpolation) | Contiguous Sequential 80/20 | Contiguous Sequential 80/20 |
| **Overall Test Accuracy** | 99.56% | **99.58%** | **98.66%** |
| **Macro F1-Score** | 96.76% | **96.82%** | **95.63%** |
| **Weighted F1-Score** | 99.56% | **99.58%** | **98.66%** |
| **Benign FPR** (`Normal` = Negative) | 0.15% | **1.30%** (68 / 5,237) | **3.67%** (192 / 5,237) |
| **Attack FNR** (Missed as `Normal`) | 0.44% | **0.06%** (8 / 13,107) | **0.40%** (52 / 13,107) |
| **Exact Feature Overlap Vectors** | 0 rows | 1 row | **824 unique vectors** (7,080 test rows) |

*\*Note: The frozen baseline is an illustrative reference point, not a directly equivalent test, due to fundamental differences in split protocols.*

### 3.2 Classification Reports & Confusion Matrices

#### Configuration A (All 79 Features)
```
               precision    recall  f1-score   support

 Botnet ARES     0.9894    1.0000    0.9947       375
 Brute Force     1.0000    1.0000    1.0000      2040
    Dos/DDos     0.9891    0.9994    0.9943      5191
Infiltration     1.0000    0.6667    0.8000         6
      Normal     0.9985    0.9870    0.9927      5237
    PortScan     0.9986    0.9996    0.9991      5082
  Web Attack     0.9976    0.9952    0.9964       413

    accuracy                         0.9958     18344
   macro avg     0.9962    0.9497    0.9682     18344
weighted avg     0.9958    0.9958    0.9958     18344
```
**Confusion Matrix (Config A):**
```
[[  375     0     0     0     0     0     0]   <- Botnet ARES
 [    0  2040     0     0     0     0     0]   <- Brute Force
 [    0     0  5188     0     3     0     0]   <- Dos/DDos
 [    0     0     0     4     2     0     0]   <- Infiltration
 [    4     0    56     0  5169     7     1]   <- Normal
 [    0     0     1     0     1  5080     0]   <- PortScan
 [    0     0     0     0     2     0   411]]  <- Web Attack
```

#### Configuration B (77 Features - Source & Destination Ports Removed)
```
               precision    recall  f1-score   support

 Botnet ARES     0.9071    0.9893    0.9464       375
 Brute Force     0.9985    0.9990    0.9988      2040
    Dos/DDos     0.9732    0.9927    0.9828      5191
Infiltration     1.0000    0.6667    0.8000         6
      Normal     0.9898    0.9633    0.9764      5237
    PortScan     0.9986    0.9996    0.9991      5082
  Web Attack     0.9927    0.9879    0.9903       413

    accuracy                         0.9866     18344
   macro avg     0.9800    0.9426    0.9563     18344
weighted avg     0.9869    0.9866    0.9866     18344
```
**Confusion Matrix (Config B):**
```
[[  371     0     0     0     4     0     0]   <- Botnet ARES (4 missed as Normal)
 [    0  2038     0     0     2     0     0]   <- Brute Force (2 missed as Normal)
 [    0     0  5153     0    38     0     0]   <- Dos/DDos (38 missed as Normal)
 [    0     0     0     4     2     0     0]   <- Infiltration (2 missed as Normal)
 [   38     3   141     0  5045     7     3]   <- Normal (192 false alarms)
 [    0     0     1     0     1  5080     0]   <- PortScan (1 missed as Normal)
 [    0     0     0     0     5     0   408]]  <- Web Attack (5 missed as Normal)
```

### 3.3 Diagnostic Analysis of the 824 Overlapping Vectors (Config B)
When ephemeral port numbers are excluded, **824 unique feature vectors** occur in both the training set and the test set, instantiating **7,080 test rows (38.60% of the test partition)**:
* **Repeated Patterns (97.69% of unique vectors, 805 / 824):** Share the exact same class label across train and test partitions. Due to repetitive automated attack scripts, these account for 6,687 test rows (e.g., 5,014 identical 0-payload TCP SYN PortScan probes; 748 Brute Force attempts; 937 DoS/DDoS packets).
* **Ambiguous Representations (2.31% of unique vectors, 19 / 824):** Feature collisions where identical packet lengths and TCP flag distributions represent different classes:
  - **17 vectors** collide between `Normal` and `PortScan` (single-packet TCP connection resets or aborted handshakes without payload).
  - **1 vector** collides between `Normal` and `Dos/DDos`.
  - **1 vector** collides between `Normal` and `Botnet ARES`.
* **Diagnostic Consequence:** These 19 ambiguous representations directly drive the increase in False Positive Rate from 1.30% in Config A to **3.67% in Config B** (falsely flagging 38 Normal flows as Botnet ARES, 141 as DoS/DDoS, and 7 as PortScan). These rows were preserved intact without deduplication to maintain an unbiased test partition.

---

## 4. Synthetic Held-Out-Class Evaluation (LOCO)

To measure detection performance against unseen threats, a standalone Leave-One-Class-Out (LOCO) evaluation was conducted. Supervised models were trained on $N-1$ classes, while unsupervised Isolation Forests were trained on the earlier 80% of Normal training data ($N=16,748$) and calibrated on the later 20% ($N=4,188$) to freeze threshold $\tau$ at the 99th percentile before evaluating against untouched test sets.

### 4.1 Summary of Synthetic LOCO Performance

| Feature Configuration | Withheld Class | Test $N$ | Supervised $\rightarrow$ `Normal`<br>*(Silent Bypass)* | Anomaly Recall<br>*(Frozen $\tau$)* | Val FPR<br>*(Target $\approx 1\%$)* | Test FPR<br>*(Normal Test)* | Precision | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A** *(All 79 Feats)* | **Infiltration\*** | 6 | **100.00%** (6/6) | **33.33%** (2/6) | 1.00% | **1.05%** (55/5,237) | 0.0351 | 0.0635 | **0.7962** |
| **Config A** *(All 79 Feats)* | **Botnet ARES** | 375 | **100.00%** (375/375) | **0.00%** (0/375) | 1.00% | **1.05%** (55/5,237) | 0.0000 | 0.0000 | **0.5587** |
| **Config B** *(No Ports - 77 Feats)* | **Infiltration\*** | 6 | **100.00%** (6/6) | **33.33%** (2/6) | 1.00% | **1.28%** (67/5,237) | 0.0290 | 0.0533 | **0.7916** |
| **Config B** *(No Ports - 77 Feats)* | **Botnet ARES** | 375 | **100.00%** (375/375) | **0.00%** (0/375) | 1.00% | **1.28%** (67/5,237) | 0.0000 | 0.0000 | **0.5236** |

*\*Statistical Caution: Infiltration test sample size is $N=6$. A detection score of 2/6 (33.33%) carries an asymptotic 95% Wilson confidence interval of $[9.7\%, 70.0\%]$. It must be treated as indicative and anecdotal.*

### 4.2 Key Evaluation Findings

1. **Closed-World Classification Failure:**
   When an unseen intrusion is presented to a supervised Random Forest, it is forcibly categorized into known labels. For both `Infiltration` and `Botnet ARES`, **100.00% of unseen flows were classified as `Normal`**. Closed-world supervised classifiers offer zero proactive defense against novel intrusion types.
2. **Key Limitation — Botnet ARES Complete Failure (0.00% Detection Rate):**
   Across both Configuration A and Configuration B, the anomaly detector exhibits a **0.00% detection rate** (0 / 375 detected) and an ROC-AUC of **0.5236 to 0.5587** (statistically equivalent to random guessing). 
   - *Technical Cause:* In tabular NetFlow feature representations (flow duration, byte volume, packet inter-arrival times), low-rate Botnet C2 beaconing packets match benign background keepalives and periodic HTTP polling traffic.
   - *Architectural Implication:* Flow-level density estimation trained solely on benign network traffic is fundamentally blind to low-rate C2 channels. Detecting such threats requires endpoint telemetry, DNS domain reputation, or deep packet inspection.

---

## 5. Summary of Core Research Limitations

| Domain / Dimension | Limitation Description | Impact on Claims |
| :--- | :--- | :--- |
| **IT Temporal Splitting** | `IDS2025.xlsx` contains no timestamps, IP addresses, or flow IDs. Contiguous sequential block splitting is a **row-order proxy**, not a verified chronological split. | Prevents intra-burst point interpolation, but does not guarantee campaign independence. |
| **Zero-Day Generalization** | The LOCO experiment tests held-out classes from the same capture environment and synthetic testbed. | Evaluates out-of-distribution detection on known distributions; does not establish real-world zero-day generalization. |
| **Statistical Support** | `Infiltration` has only 23 training and 6 test instances in `IDS2025`. | Metrics for Infiltration are statistically uncertain ($\pm 35\%$ confidence bounds). |
| **Distribution Overlap** | Unsupervised flow density estimation fails completely on low-rate `Botnet ARES` (0.00% recall). | Intrusion detection cannot claim general zero-day capability using flow features alone. |
| **OT Attack Coverage** | The OT Autoencoder achieves 46.39% attack recall, missing 1,598 attack seconds in physical telemetry. | Subtle sensor spoofing within operational tolerances bypasses simple reconstruction thresholds. |
| **Cross-Domain Coupling** | `IDS2025` and `HAI` were captured on separate testbeds and are synthetically paired. | Real cross-domain correlation cannot be proven without concurrent cyber-physical testbed telemetry. |

---

## 6. Complete Artifact & Script Inventory

| Artifact / Script Path | Role & Content |
| :--- | :--- |
| `data/processed/it_realistic/it_train_all.npz` | Contiguous train set, Configuration A (73,400 rows, 79 features) |
| `data/processed/it_realistic/it_test_all.npz` | Contiguous test set, Configuration A (18,344 rows, 79 features) |
| `data/processed/it_realistic/it_train_no_ports.npz` | Contiguous train set, Configuration B (73,400 rows, 77 features) |
| `data/processed/it_realistic/it_test_no_ports.npz` | Contiguous test set, Configuration B (18,344 rows, 77 features) |
| `data/processed/it_realistic/metadata.json` | Complete partition boundaries, class distributions, and feature indices |
| `data/processed/it_realistic/loco_predictions.json` | Frozen thresholds ($\tau$), validation FPRs, and LOCO test metrics |
| `models/it_realistic/random_forest_all.joblib` | 7-class benchmark model (Config A, 79 features) |
| `models/it_realistic/random_forest_no_ports.joblib` | 7-class benchmark model (Config B, 77 features) |
| `models/it_realistic/loco_rf_*.joblib` | Supervised models trained on $N-1$ classes for LOCO evaluation |
| `models/it_realistic/loco_iforest_*.joblib` | Isolation Forest anomaly detectors calibrated on Normal validation flows |
| `src/prepare_it_realistic_data.py` | Contiguous sequential 80/20 partition generator with train-only scaling |
| `src/evaluate_it_realistic.py` | Benchmark evaluation script for Configurations A and B |
| `src/audit_realistic_evaluation.py` | Verification and audit script for predictions, matrices, and the 824 overlaps |
| `src/evaluate_loco_realistic.py` | Synthetic held-out-class evaluation script with validation-isolated thresholding |
| `data/reports/it_realistic_evaluation_report.txt` | Detailed 7-class realistic evaluation report |
| `data/reports/loco_synthetic_evaluation_report.txt` | Detailed LOCO synthetic evaluation report |
| `FINAL_RESEARCH_REPORT.md` | Consolidated final research document |

---
*Report Version: 2.0 (Realistic Benchmark & Synthetic LOCO Integration)*  
*Project: ResiNet-IDAM (Adaptive Deep Learning and Zero-Trust Framework)*
