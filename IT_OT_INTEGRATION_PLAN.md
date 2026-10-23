# ResiNet-IDAM: IT-OT Integration Model Implementation Plan

## Adaptive Deep Learning and Zero-Trust Framework for Detecting Zero-Day Intrusions in IT-OT Hybrid Critical Infrastructures

---

## 1. Executive Summary & Design Principles

ResiNet-IDAM aims to protect IT-OT hybrid critical infrastructures by combining IT network intrusion detection with OT anomaly detection, reinforced by Zero Trust principles and cross-domain event correlation.

### System Classification: Prototype Status
> [!IMPORTANT]
> Until realistic, empirically captured cross-domain telemetry is available, the integration module is explicitly designated and evaluated as a **research prototype**, not a field-validated zero-day detection system. Experimental performance targets are benchmarks for iterative development, not guaranteed outcomes.

### Primary Research Limitation: Synthetic Cross-Domain Pairing
> [!WARNING]
> **Synthetic Pairing Limitation:** In the current implementation, IT network flows (from `IDS2025`) and OT sensor records (from `HAI`) originate from completely independent sources and are synthetically paired. They do not share genuine temporal synchronization, unified host/network topologies, or causal attack sequences. 
> **Scientific Implication:** Real cross-domain intrusion detection cannot be definitively validated until empirical, genuinely time-aligned IT and OT telemetry (e.g., from concurrent hybrid testbeds) is collected and evaluated. All current integration findings must be interpreted strictly within the context of a synthetic benchmark.

### Core Architectural Principles
1. **Preserve Validated Baselines:** Retain the existing IT Random Forest and OT Autoencoder/Isolation Forest baseline models as specialized domain detectors.
2. **Prerequisites First:** Event alignment, strict validation isolation, and verifiable ground-truth labels are mandatory prerequisites prior to training any fusion model.
3. **Risk Adjustment, Not Unconditional Suppression:** Do not suppress isolated OT alerts simply because there is no IT precursor. In real environments, direct physical tampering, insider access, or unmonitored fieldbus attacks leave zero IT footprint; furthermore, the baseline OT Autoencoder currently misses 1,598 attacks. Requiring IT corroboration for every alert would dangerously escalate false negatives.
4. **Transparent Baselines Before Complex Networks:** Formulate and evaluate a transparent mathematical fusion equation before introducing multi-layer perceptrons (MLP).
5. **Rigorous Zero-Day Testing:** A supervised 7-class classifier will forcibly assign an unseen attack to one of its known classes. Zero-day evaluation must employ anomaly detection mechanisms (e.g., Autoencoder reconstruction or prediction uncertainty) and all synthetic withheld-class experiments must be transparently labeled as synthetic.

---

## 2. Data Verification: Current IT vs. OT Assets

Both datasets have been verified in `data/processed/`. Their profiles, semantics, and representations are summarized below:

| Dimension | IT Intrusion Telemetry (`data/processed/it/`) | OT Industrial Telemetry (`data/processed/`) |
|---|---|---|
| **Primary Files** | `it_train.npz`, `it_test.npz`, `metadata.json` | `hai-train-scaled.csv`, `hai-test-scaled.csv`, `hai-train-features.csv` |
| **Data Nature** | Tabular network flow summaries (NetFlow/IPFIX/CICFlow style) | Continuous 1 Hz physical sensor & actuator time series |
| **Training Size** | **73,395 records** (80% split) | **280,800 records** (1-second intervals across 4 days) |
| **Testing Size** | **18,349 records** (20% split) | **54,000 records** (15-hour test window) |
| **Feature Count** | **79 numeric features** (IAT, packet lengths, flow rates, ports) | **66 physical features** (valves `FCV`, flow `FT`, pressure `PIT`, temp `TIT`) |
| **Target Variable** | `newLabel` (7 discrete classes) | `label` (Binary: 0 = Normal, 1 = Attack) |
| **Class Distribution (Test)**| • Normal: 5,235<br>• Dos/DDos: 5,198<br>• PortScan: 5,082<br>• Brute Force: 2,040<br>• Web Attack: 413<br>• Botnet ARES: 375<br>• Infiltration: 6 | • Normal: 51,019 (94.48%)<br>• Attack: 2,981 (5.52%) |
| **Preprocessing Applied** | Exact deduplication (-86 rows), median imputation, StandardScaler | Constant column removal (-20 cols), time sorting, StandardScaler |

---

## 3. Baseline Performance & Benchmark Accuracies

The existing standalone baseline models achieved the following documented metrics:

### 3.1 IT Detection Baseline (Random Forest - 200 Trees, Balanced)
* **Accuracy:** **`99.56%`** (18,269 / 18,349 correct test predictions)
* **Macro F1-Score:** **`96.76%`** | **Weighted F1-Score:** **`99.56%`**
* **Key Vulnerabilities:**
  * Severe class imbalance in `Infiltration` (only 29 total instances, 6 in test), yielding only **66.67% recall** (4/6 detected).
  * Evaluated under a closed-world setting: cannot identify novel zero-day attacks without anomaly-based extensions.

### 3.2 OT Anomaly Detection Baselines (HAI Industrial Testbed - 54,000 Test Records)
* **Isolation Forest Baseline:**
  * **Accuracy:** **`65.86%`** (35,564 / 54,000 correct)
  * **Attack Precision:** `10.49%` | **Attack Recall:** `68.80%` (2,051 / 2,981 attacks caught)
  * **Attack F1-Score:** `18.20%`
  * **Errors:** `17,506` False Positives, `930` False Negatives
* **Deep Learning Autoencoder (Reconstruction Threshold = 0.82195, 99th percentile):**
  * **Accuracy:** **`92.27%`** (49,824 / 54,000 correct)
  * **Attack Precision:** `34.92%` | **Attack Recall:** `46.39%` (1,383 / 2,981 attacks caught)
  * **Attack F1-Score:** `39.84%`
  * **Errors:** `2,578` False Positives, `1,598` False Negatives

---

## 4. Multi-Channel Triage Architecture & Risk Formulation

### 4.1 Separate Domain Risk Outputs
To prevent target leakage and avoid single-point alert suppression, the architecture produces **three distinct, uncorrupted risk channels** derived strictly from real telemetry features:

1. **IT Attack Probability ($R_{\text{IT}}(t) = p_{\text{attack}}(t) \in [0, 1]$):**
   Derived from the calibrated class probability distribution of the IT Random Forest:
   $$R_{\text{IT}}(t) = p_{\text{attack}}(t) = 1 - P_{\text{IT}}(\text{Normal}, t) = \sum_{c \neq \text{Normal}} P_{\text{IT}}(c, t)$$
   Monitored by the Enterprise SOC. Triggers an alert when $R_{\text{IT}}(t) \ge \tau_{\text{IT}}$.

2. **OT Physical Anomaly Score ($R_{\text{OT}}(t) = S_{\text{OT}}(t) \in [0, 1]$):**
   Derived from the Autoencoder mean squared reconstruction error $\text{MSE}(t)$ using a calibrated sigmoid function centered at the validation threshold $\theta$:
   $$R_{\text{OT}}(t) = S_{\text{OT}}(t) = \frac{1}{1 + \exp\left(-\gamma \cdot \left(\text{MSE}(t) - \theta\right)\right)}$$
   where $\gamma$ is fit on the validation split. Monitored by Plant Operations. Triggers an alert when $R_{\text{OT}}(t) \ge \tau_{\text{OT}}$.

3. **Cyber-Physical Correlation Risk ($R_{\text{CP}}(t) \in [0, 1]$):**
   Trained strictly on empirical telemetry features ($p_{\text{attack}}$, $S_{\text{OT}}$, and their interaction) with **all synthetic Zero-Trust shortcuts removed**:
   $$R_{\text{CP}}(t) = \sigma\Big(\alpha \cdot p_{\text{attack}}(t) + \beta \cdot S_{\text{OT}}(t) + \mu \cdot \left[p_{\text{attack}}(t) \cdot S_{\text{OT}}(t)\right] + \text{intercept}\Big)$$
   Evaluates cross-domain escalation. Triggers a Tier-4 Critical Alert when $R_{\text{CP}}(t) \ge \tau_{\text{CP}}$.

### 4.2 Multi-Tier Incident Triage Matrix
Alerts are routed to their respective operational owners without unconditional suppression:
* **Tier 4 (Critical Cyber-Physical Incident):** $R_{\text{CP}} \ge \tau_{\text{CP}}$ or $(R_{\text{IT}} \ge \tau_{\text{IT}} \land R_{\text{OT}} \ge \tau_{\text{OT}})$
  * Coordinated attack affecting both enterprise network and industrial actuators/sensors.
* **Tier 2 (Process Alert - Plant Engineering):** $R_{\text{OT}} \ge \tau_{\text{OT}} \land R_{\text{CP}} < \tau_{\text{CP}}$
  * Physical anomaly detected (e.g. Scenario D - Isolated OT attack or equipment degradation). Dispatched to engineering without requiring IT corroboration.
* **Tier 2 (Enterprise Alert - SOC Investigation):** $R_{\text{IT}} \ge \tau_{\text{IT}} \land R_{\text{CP}} < \tau_{\text{CP}}$
  * IT intrusion attempt (e.g. Scenario C - Isolated DoS/Web Attack) contained at network perimeter before reaching industrial controllers.
* **Tier 1 (Normal Operation):** All channels below thresholds.

---

## 5. Realistic Zero-Day Evaluation Protocol

A standard 7-class Random Forest will always assign an unseen attack to one of its 7 trained classes, often with high confidence. To substantiate zero-day intrusion detection claims, the evaluation framework implements:

1. **Anomaly-Based Detection Layer for IT:**
   * Evaluate out-of-distribution (OOD) indicators: prediction entropy $H(P_{\text{IT}})$, maximum softmax probability shortfall, or a dedicated IT Autoencoder / One-Class SVM trained strictly on `Normal` flows.
2. **Leave-One-Class-Out (LOCO) Protocol:**
   * Systematically withhold one full attack class (e.g., `Infiltration`, `Botnet ARES`, or `Web Attack`) during supervised training.
   * Evaluate whether the anomaly layer and the fusion engine flag the withheld class as anomalous rather than misclassifying it as normal or mislabeling it under another attack class.
3. **Explicit Labeling of Synthetic Scenarios:**
   * Because `IDS2025` and `HAI` were captured on separate testbeds without unified timestamps, joint multi-stage scenarios are synthetically synchronized.
   * All experiments using stitched or aligned scenarios must be explicitly designated: **"Synthetic Cyber-Physical Evaluation Benchmark (Version 1.0)"**.

---

## 6. Corrected Implementation Order & Workflow

```
[Step 1: Artifact & Schema Verification]
  - Audit saved model paths, scalers, imputers, and feature ordering
  - Verify OT timestamp continuity and IT feature dictionary
            │
            ▼
[Step 2: Event Alignment & Ground-Truth Scenarios]
  - Establish whether genuine cross-domain links exist in raw telemetry
  - If not, formally synthesize and label 4 benchmark evaluation scenarios
            │
            ▼
[Step 3: Validation-Isolated Score Calibration]
  - Split validation partitions from both IT and OT pipelines (strictly isolated from test)
  - Fit Platt / isotonic calibration for IT and sigmoid temperature for OT
            │
            ▼
[Step 4: Transparent Fusion Baseline Evaluation]
  - Optimize coefficients [α, β, λ, μ] on validation data
  - Benchmark fusion model directly against IT-only and OT-only baselines
            │
            ▼
[Step 5: Multi-Scenario Generalization & Latency Testing]
  - Evaluate Scenario A, B, C, D on a held-out test scenario partition
  - Measure False Positive Rate, Attack Recall, and Detection Delay (MTTD)
            │
            ▼
[Step 6: LOCO Zero-Day & Advanced Neural Fusion (Optional)]
  - Execute Leave-One-Class-Out zero-day testing on the anomaly detection layer
  - Incorporate Zero-Trust contextual signals
  - Compare performance against transparent fusion before exploring multi-layer neural networks
```

---

## 7. Detailed Step-by-Step Task Descriptions

### Step 1: Verify Existing Artifacts and Feature Schemas
* Verify integrity of existing artifacts:
  * IT: `models/it/random_forest.joblib`, `models/it/preprocessing.joblib`, `models/it/label_encoder.joblib`, `data/processed/it/metadata.json`.
  * OT: `models/ot_autoencoder.keras`, `models/isolation_forest.joblib`, `models/standard_scaler.joblib`, `models/autoencoder_threshold.json`.
* Validate that test feature columns strictly match training column names, ordering, and data types.
* Verify OT timestamp formats and record sampling frequency (1.0 Hz verification).

### Step 2: Establish Event Alignment and Scenarios
* Audit whether shared flow keys, timestamps, or IP subnets link `IDS2025` and `HAI`.
* In their absence, construct a formally documented **Synthetic Multi-Stage Scenario Generator**:
  * **Scenario A (Correlated Attack):** IT Brute Force / PortScan preceding an OT setpoint change (`x1001_05_SETPOINT_OUT`) within window $\Delta T = 60\text{s}$.
  * **Scenario B (Stealth Physical Zero-Day):** Low-rate IT Infiltration combined with gradual OT pressure/vibration drift (`P1_PIT01`, `P2_VIBTR01`).
  * **Scenario C (Isolated IT Cyber Threat):** High-rate IT DoS with normal OT operation (evaluates false plant trip mitigation).
  * **Scenario D (Isolated OT Physical Anomaly):** Mechanical sensor fault with zero IT network footprint (evaluates risk adjustment without false suppression).
* Partition scenarios into dedicated **Scenario-Train/Validation** and **Scenario-Test** sets to evaluate generalization on unseen scenario permutations.

### Step 3: Calibrate Scores on Isolated Validation Data
* Reserve 20% of training data exclusively as a cross-domain validation split.
* Train calibration mappings on this validation partition:
  * IT: Fit probability calibrator to produce accurate $P_{\text{IT}}(\text{Normal})$ and $p_{\text{attack}} = 1 - P_{\text{IT}}(\text{Normal})$.
  * OT: Fit scaling parameter $\gamma$ on validation reconstruction errors relative to $\theta = 0.82195$.
* Ensure test data remains completely untouched during this phase.

### Step 4: Implement & Evaluate Transparent Fusion Baseline
* Implement the transparent fusion equation:
  $$R(t) = \alpha p_{\text{attack}}(t) + \beta S_{\text{OT}}(t) + \lambda ZT(t) + \mu p_{\text{attack}}(t) S_{\text{OT}}(t)$$
* Solve for optimal weights $[\alpha, \beta, \lambda, \mu]$ via logistic regression or ROC optimization on the validation set.
* Compare against standalone baselines:
  * Baseline 1: IT-only detector ($R(t) = p_{\text{attack}}(t)$).
  * Baseline 2: OT-only detector ($R(t) = S_{\text{OT}}(t)$).
  * Integration: Joint Risk Model $R(t)$.

### Step 5: Comprehensive Scenario & Latency Evaluation
* Run evaluation on the held-out scenario test set.
* Report the full metric suite:
  * **Detection Recall:** Percentage of cyber-physical attack sequences detected.
  * **False Positive Rate:** Normal baseline false-alarm rate per hour.
  * **Detection Delay (MTTD):** Time difference between initial IT precursor detection and physical impact warning.
  * **Risk-Adjustment Verification:** Confirm that isolated OT anomalies (Scenario D) trigger appropriate Tier-2 process alerts rather than being suppressed.

### Step 6: LOCO Zero-Day Testing & Zero Trust Context
* Execute synthetic Leave-One-Class-Out (LOCO) tests:
  * Exclude `Infiltration` from IT training; verify if the anomaly layer identifies it as abnormal.
  * Exclude specific attack intervals from OT training; verify autoencoder reconstruction error response.
* Introduce contextual Zero-Trust signals ($ZT(t)$) and evaluate incremental benefit over raw telemetry fusion.
* Only if the transparent linear-synergy fusion model demonstrates a performance ceiling, compare it against a Multi-Layer Perceptron (MLP) late-fusion network.

---

## 8. Milestone Deliverables & Artifact Map

| Milestone | Target Script / Artifact | Description |
|---|---|---|
| **M1: Schema Audit** | `src/integration/verify_schemas.py` | Validates model files, preprocessing pipelines, and column order consistency |
| **M2: Scenario Generator** | `src/integration/generate_scenarios.py` | Synthesizes and labels four synchronized cyber-physical test scenarios |
| **M3: Score Calibrator** | `src/integration/calibrate_scores.py` | Generates calibrated $p_{\text{attack}}$ and $S_{\text{OT}}$ using isolated validation splits |
| **M4: Fusion Model** | `src/integration/train_fusion_model.py` | Optimizes $[\alpha, \beta, \lambda, \mu]$ and evaluates the transparent fusion equation |
| **M5: Evaluation Report** | `data/reports/integration_report.txt` | Documents per-scenario detection, false alarm rates, and detection latency |
| **M6: Zero-Day Audit** | `src/integration/evaluate_loco.py` | Assesses LOCO detection on withheld attack classes using anomaly estimators |

---
*Document Version: 2.0 (Revised per Evaluation Guidelines)*  
*Project: ResiNet-IDAM*
