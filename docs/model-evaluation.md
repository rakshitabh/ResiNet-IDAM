# Comprehensive Model Evaluation and Detection Benchmarks

## 1. Executive Summary
This document summarizes the empirical evaluation across IT flow detection, OT physical telemetry anomaly detection, and synthetic zero-day (Leave-One-Class-Out) experiments in ResiNet-IDAM v2.0.

> [!IMPORTANT]
> **Core Research Limitations**:
> 1. **Dataset-Specific Results**: All metrics reflect the experimental characteristics of the IDS2025 and HAI testbeds and do not imply universal real-world performance.
> 2. **Synthetic Zero-Day Evaluation**: Held-out-class experiments are synthetic laboratory approximations. This framework is a *synthetic held-out-class evaluation and anomaly-based research prototype for novel intrusion detection*, not a production zero-day defense.
> 3. **Asynchronous Telemetry**: IT and OT datasets originate from separate testbeds and are synthetically paired for multi-channel correlation demonstration.
> 4. **Decision-Support Boundary**: The framework is an analyst decision-support system. It suggests triage actions and **never executes automatic industrial control actions**.

---

## 2. IT Flow Detection Benchmark (IDS2025)

The IT detection baseline uses a balanced Random Forest classifier (`n_estimators=200, class_weight='balanced', random_state=42`) evaluated under contiguous chronological 80/20 train/test partitioning to prevent burst leakage.

### Configurations
* **Configuration A (All 79 Features)**: Full NetFlow feature set including ephemeral source/destination ports.
* **Configuration B (77 Features - No Ports)**: Source and destination port features removed to audit against shortcut learning.

### Performance Comparison
| Metric | Configuration A (79 Features) | Configuration B (77 Features - No Ports) | Notes |
|---|---:|---:|---|
| **Accuracy** | 99.58% | 98.66% | Realistic temporal split |
| **Macro F1** | 96.82% | 95.63% | Balanced across all 7 classes |
| **Attack False-Negative Rate (FNR)** | 0.06% | 0.40% | Missed attacks as normal |
| **Benign False-Positive Rate (FPR)** | 1.30% | 3.67% | Benign flows flagged as attack |

> [!NOTE]
> **Shortcut Learning Audit**: The 79-feature configuration achieves higher benchmark scores partly by learning ephemeral port assignments. Configuration B evaluates true behavioral flow dynamics and serves as the robustness audit.

---

## 3. OT Anomaly Detection Benchmark (HAI Industrial Telemetry)

Evaluated on the Hardware-in-the-Loop (HAI) testbed over 66 synchronized physical sensor and actuator signals.

### Autoencoder Architecture
```text
Input (66) -> Dense (32, ReLU) -> Bottleneck (16, ReLU) -> Dense (32, ReLU) -> Output (66, Linear)
```
Trained on chronological normal data with threshold calibrated at the 99th percentile of validation reconstruction error ($\tau \approx 0.822$).

### Model Comparison: Autoencoder vs. Isolation Forest
| Metric | Autoencoder (Primary) | Isolation Forest (Comparison) | Operational Implication |
|---|---:|---:|---|
| **Precision** | 34.92% | 10.49% | Autoencoder produces ~6.8x fewer false alarms |
| **Recall** | 46.39% | 68.80% | Isolation Forest catches more attacks but floods alerts |
| **F1-Score** | 39.84% | 18.20% | Autoencoder offers superior operational balance |
| **False Positives** | 2,578 | 17,506 | High FPR in IF is disruptive to plant operators |
| **False Negatives** | 1,598 | 930 | Missed anomaly intervals |

**Decision**: The Autoencoder serves as the primary OT anomaly detector to minimize operational alarm fatigue, with Isolation Forest available as a high-recall secondary auditor.

---

## 4. Synthetic Zero-Day (LOCO) Evaluation

To assess novel intrusion behavior, supervised models were trained withholding one attack class entirely, coupled with an unsupervised anomaly detector calibrated exclusively on normal validation traffic.

### Empirical Findings
1. **Botnet ARES (Held Out)**:
   * **Supervised Classifier**: 0% detected (100% misclassified as Normal). Supervised closed-world classifiers cannot identify unseen classes.
   * **Unsupervised Anomaly Detector**: 0% anomaly recall (scores remained below the frozen threshold). Botnet ARES command-and-control flows closely mimic benign network patterns in aggregate flow statistics.
2. **Infiltration (Held Out)**:
   * **Supervised Classifier**: 0% detected (100% misclassified as PortScan).
   * **Unsupervised Anomaly Detector**: 66.7% anomaly recall (4 of 6 test samples detected). *Treated with high uncertainty due to small test sample count ($N=6$).*

---

## 5. Research Metrics vs. Operational Metrics

To transition from pure academic evaluation to a security operations console, metrics must be divided into laboratory and production operational indicators:

| Category | Metric | Definition & Purpose |
|---|---|---|
| **Research** | Macro F1 | Unweighted harmonic mean of precision and recall across all classes |
| **Research** | PR-AUC / ROC-AUC | Threshold-independent discrimination capability on imbalanced sets |
| **Research** | Per-Class Recall | Diagnostic breakdown identifying vulnerable attack classes |
| **Operational** | False-Positive Rate (FPR) | Fraction of benign events flagged as alerts (directly impacts alert fatigue) |
| **Operational** | False-Negative Rate (FNR) | Critical security risk: percentage of malicious flows that evade detection |
| **Operational** | Alert Volume / Hour | Real-time queue volume indicating operational analyst workload |
| **Operational** | Detection Delay | Elapsed time between event timestamp and alert creation |
| **Operational** | Model Availability | Health heartbeat confirming detector artifacts and engines are online |
| **Operational** | Data Freshness | Timeliness of incoming telemetry relative to expected ingest intervals |
| **Operational** | Feature Drift | Divergence between inference inputs and training baseline distributions |
