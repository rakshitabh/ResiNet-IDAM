# ResiNet-IDAM v2.0: Complete Technical Architecture & Evaluation Report

**Document Version:** 2.0.0  
**Status:** Verified & Released  
**Operating Environment:** Industrial Control Systems (ICS) / Operational Technology (OT) & NetFlow IT  
**Safety Scope:** Human-in-the-Loop Decision Support Only (Zero Autonomous Actuation)

---

## 1. System Architecture & End-to-End Pipeline

ResiNet-IDAM v2.0 provides unified, explainable anomaly detection and Zero-Trust risk assessment across dual industrial environments (IT Network Flows and OT Cyber-Physical SCADA Streams).

```
+---------------------------------------------------------------------------------------------------+
|                                      DATA INGESTION PIPELINE                                      |
|   OT SCADA Telemetry (66 Sensor/Actuator Tags)       |    IT NetFlow Telemetry (79 Flow Features) |
+------------------------------------------------------+--------------------------------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------------------------+
|                                  FEATURE SCALING & NORMALIZATION                                  |
|   OT: StandardScaler (models/standard_scaler.joblib)  |    IT: RobustScaler (79-dim flow contract) |
|   Formula: z_j = (x_j - mu_j) / sigma_j              |    Provenance: SHA-256 Verified Artifacts  |
+---------------------------------------------------------------------------------------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------------------------+
|                                     DUAL DETECTION ENGINE                                         |
|   1. Residual Autoencoder (Skip Connections)         |    1. Residual / Attention Autoencoder     |
|   2. Isolation Forest (Calibrated Contamination)     |    2. Isolation Forest                     |
+---------------------------------------------------------------------------------------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------------------------+
|                                CALIBRATED MODEL FUSION (PHASE 5)                                  |
|   Sigmoid Calibration: P_m = 1 / (1 + exp(-A_m * (s_m - B_m)))                                    |
|   OT Fusion: P_fused = 0.45 * P_AE + 0.55 * P_IF                                                  |
|   Output: Unified Probability Score in [0, 1]                                                     |
+---------------------------------------------------------------------------------------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------------------------+
|                              ADAPTIVE THRESHOLD & DRIFT MONITOR                                   |
|   Adaptive Threshold: T(t) = mu(t) + k * sigma(t) (Anti-Poisoning Quarantine Buffer)               |
|   Concept Drift: Block Page-Hinkley S_k = max(0, S_{k-1} + (m_k - bar{mu} - delta)) & ADWIN       |
+---------------------------------------------------------------------------------------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------------------------+
|                             EXPLAINABILITY & ROOT-CAUSE ATTRIBUTION                               |
|   Feature Residuals: e_j = (x_j - hat{x}_j)^2                                                     |
|   Ranking: Top-K Deviating Sensors (e.g., P1_FCV01D, P1_FIT01) + Fusion Weight Breakdown           |
+---------------------------------------------------------------------------------------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------------------------+
|                                ZERO-TRUST CONTEXTUAL RISK ENGINE                                  |
|   Risk Score = 100 * [0.25*Anomaly + 0.30*Behavior + 0.15*IdentityUntrust +                       |
|                       0.10*DeviceUntrust + 0.20*AssetCriticality]                                  |
|   Transparent Rules: R1 (Coordinated Attack), R2 (Compromised Trust), R3 (Elevated Risk)          |
+---------------------------------------------------------------------------------------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------------------------+
|                           SECOPS OPERATOR CONSOLE (DECISION SUPPORT)                              |
|   UI Dashboard: Plotly XAI Bar Charts | Real-Time Drift Panel | Incident Triage Buttons           |
|   Status Lifecycle: NEW -> ACKNOWLEDGED -> ESCALATED / DISMISSED                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Feature Flow, Data Contracts & Preprocessing Parameters

### 2.1 Feature Contracts

| Parameter | OT Cyber-Physical Domain | IT Network Flow Domain |
|---|---|---|
| **Primary Dataset** | HAI 22.04 (Hardware-in-the-Loop) | CIC-IDS2025 / NetFlow v9 |
| **Input Feature Count** | **66 features** | **79 features** |
| **Subsystems Monitored** | Primary Loop (P1), Secondary Loop (P2), Turbine (P3/P4) | Ingress/Egress Routers, SCADA Gateways, Workstations |
| **Test-Constant Columns** | **7 features preserved** (`P1_PP04`, `P1_SOL01D`, `P1_SOL03D`, `P2_AutoSD`, `P2_Emerg`, `P2_TripEx`, `P3_LH01`) | Constant flow attributes zero-filled |
| **Sampling Rate** | 1.0 Hz (1 sample per second) | Flow-duration triggered (event-driven) |

### 2.2 Normalization Algorithm & Mathematics

$$\mu_j = \frac{1}{N} \sum_{i=1}^N x_{i,j}, \quad \sigma_j = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_{i,j} - \mu_j)^2 + \epsilon}$$

$$z_{i,j} = \frac{x_{i,j} - \mu_j}{\sigma_j}$$

- **Safety Constraint**: If $\sigma_j = 0$ (constant feature), $\sigma_j \leftarrow 1.0$, mapping $z_{i,j} = 0.0$ to prevent numerical division-by-zero.
- **Persistence**: `models/standard_scaler.joblib` with SHA-256 provenance integrity hash: `f2c6e57930953d4c3a50d248bdfb1fe164f9b8287ff27aa51cbb580132338d38`.

---

## 3. Core Algorithms, Mathematical Formulations & Parameters

### 3.1 Model Architectures

#### A. OT Residual Autoencoder (NumpyResidualAutoencoder)
- **Bottleneck Architecture**: $66 \to 32 \to 16 \to 32 \to 66$.
- **Skip Connections**: Deep residual connections add the encoder representation to the decoder mirror layer:
  $$h_{\text{dec}} = \text{ReLU}(W_{\text{dec}} h_{\text{latent}} + b_{\text{dec}}) + h_{\text{enc}}$$
- **Reconstruction Error Calculation**:
  $$e(t) = \frac{1}{D} \sum_{j=1}^D (x_j(t) - \hat{x}_j(t))^2$$

#### B. Causal Sequence Window Scorer
- **Causal Formulation** (strictly backward-looking, zero future data leakage):
  $$S(t) = \frac{1}{L} \sum_{k=0}^{L-1} e(t - k), \quad L = 5$$

#### C. Isolation Forest (Unsupervised Tree Ensemble)
- **Subsampling**: 256 samples per tree.
- **Number of Estimators**: 100 isolation trees.
- **Anomaly Path Length Formulation**:
  $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}, \quad c(n) = 2 \ln(n - 1) + 0.5772156649 - \frac{2(n - 1)}{n}$$

---

### 3.2 Adaptive Thresholding with Poisoning Safeguards (Phase 3)

$$T(t) = \mu_W(t) + k \cdot \sigma_W(t)$$

| Parameter | Value | Operational Purpose |
|---|---|---|
| **Sliding Window Size ($W$)** | $500$ samples | Tracks moving baseline statistics without excessive memory. |
| **Multiplier ($k$)** | $3.0$ | Calibrated to maintain $<1\%$ false-alarm rate on normal operations. |
| **Safety Floor ($T_{\min}$)** | $0.0899$ | Prevents threshold collapse to zero during extended plant idling. |
| **Max Relative Change ($\delta_{\max}$)** | $0.05$ ($5\%$) | Rate-of-change clamp preventing rapid threshold jumps. |
| **Anti-Poisoning Buffer Rule** | $e(t) > T(t) \implies \text{Quarantine}$ | Anomalous and attack samples are **excluded** from the moving window buffer, preventing adversarial threshold poisoning. |

---

### 3.3 Concept Drift Monitoring (Phase 4)

#### Block Page-Hinkley Cumulative Sum Detector
Tracks slow physical wear and operational shifts in heavy-tailed OT residual streams:

$$m_k = \frac{1}{B} \sum_{i=(k-1)B + 1}^{kB} e(i), \quad B = 50$$

$$S_k = \max(0, S_{k-1} + (m_k - \bar{\mu} - \delta))$$

$$\text{Condition: } S_k > \lambda \implies \text{DRIFT\_CONFIRMED} \to \text{REVIEW\_REQUIRED}$$

| Parameter | OT Value | Description |
|---|---|---|
| **Block Size ($B$)** | $50$ samples | Averages out high-frequency sensor noise. |
| **Allowance ($\delta$)** | $0.005$ | Minimum drift magnitude considered significant. |
| **Threshold ($\lambda$)** | $50.0$ | Cumulative sum barrier before raising an operational review alert. |

---

### 3.4 Calibrated Model Fusion & Zero-Trust Alignment (Phase 5)

Platt Sigmoid Calibration maps uncalibrated detector scores to true empirical probabilities:

$$P_m(y=1 \mid s) = \frac{1}{1 + \exp\left(-A_m \cdot (s - B_m)\right)}$$

$$P_{\text{fused}} = w_{\text{AE}} \cdot P_{\text{AE}} + w_{\text{IF}} \cdot P_{\text{IF}}$$

| Model Parameter | OT Domain Value | IT Domain Value |
|---|:---:|:---:|
| **$A_{\text{AE}}$ (Sigmoid Slope)** | $12.5$ | $15.0$ |
| **$B_{\text{AE}}$ (Sigmoid Intercept)** | $0.0899$ | $0.0512$ |
| **$A_{\text{IF}}$ (Sigmoid Slope)** | $18.0$ | $12.0$ |
| **$B_{\text{IF}}$ (Sigmoid Intercept)** | $0.5000$ | $0.5000$ |
| **Fusion Weight $w_{\text{AE}}$** | **$0.45$** | **$0.50$** |
| **Fusion Weight $w_{\text{IF}}$** | **$0.55$** | **$0.50$** |
| **Decision Boundary** | **$0.50$** | **$0.50$** |

---

### 3.5 Explainability & Root-Cause Attribution (Phase 6)

For every anomalous event, individual feature reconstruction errors are calculated and sorted:

$$e_j(t) = \left(x_j(t) - \hat{x}_j(t)\right)^2, \quad \text{Attribution Proportion } \pi_j = \frac{e_j(t)}{\sum_{k=1}^D e_k(t)}$$

- **Output**: Top 3 root-cause physical sensors (e.g., `P1_FCV01D`: $44.2\%$, `P1_FIT01`: $28.1\%$, `P1_LIT01`: $12.4\%$) displayed as interactive Plotly bar charts in the SecOps console.

---

### 3.6 Zero-Trust Risk Engine & Access Rules

$$\text{Risk Score} = 100 \times \sum_{i=1}^5 w_i \cdot c_i$$

| Risk Factor | Weight ($w_i$) | Component Formula ($c_i$) | Operational Meaning |
|---|:---:|---|---|
| **Behavior Deviation** | $0.30$ ($30\%$) | $c_{\text{beh}} \in [0, 1]$ | Distance from historical operational profile |
| **Anomaly Severity** | $0.25$ ($25\%$) | $P_{\text{fused}} \in [0, 1]$ | Calibrated anomaly probability from models |
| **Asset Criticality** | $0.20$ ($20\%$) | $c_{\text{crit}} \in [0, 1]$ | Importance of target physical asset (e.g. Turbine = 0.95) |
| **Identity Untrust** | $0.15$ ($15\%$) | $1.0 - \text{IdentityTrust}$ | Credentials, role validity, and MFA status |
| **Device Untrust** | $0.10$ ($10\%$) | $1.0 - \text{DeviceTrust}$ | Device posture, MAC authentication, firmware status |

#### Deterministic Policy Decision Rules

| Rule ID | Logic Trigger Condition | Verdict | Policy Recommendation |
|---|---|:---:|---|
| **R1** | $\text{Anomaly} \ge 0.80 \land \text{Behavior} \ge 0.65 \land \text{Asset} \ge 0.60$ | **DENY** | `RECOMMEND_CONTAINMENT` |
| **R2** | $\text{IdentityTrust} < 0.40 \land \text{DeviceTrust} < 0.40 \land \text{Risk} \ge 60$ | **DENY** | `REVOKE_AND_ISOLATE` |
| **R3** | $\text{Risk} \ge 60 \lor \text{IdentityTrust} < 0.60 \lor \text{DeviceTrust} < 0.60$ | **CHALLENGE** | `STEP_UP_AUTHENTICATION` |
| **Default** | $\text{Risk} < 30 \land \text{Trust} \ge 0.80$ | **ALLOW** | `RECORD_AND_MONITOR` |

---

## 4. Benchmark Results & Accuracy Metrics

### 4.1 Cumulative Ablation Study (E0 through E5)

*Tested on the full labeled HAI OT test benchmark ($N=54,000$ samples):*

| Config | Architecture / Pipeline Stage | Precision | Recall | F1-Score | FPR | Key Improvement Notes |
|---|---|:---:|:---:|:---:|:---:|---|
| **E0** | Baseline Autoencoder | 34.92% | 46.37% | 39.84% | 5.05% | Static reconstruction threshold baseline |
| **E1** | Residual Autoencoder | 35.80% | 45.61% | 40.12% | 4.78% | Residual skip-connections stabilize gradients |
| **E2** | Adaptive Thresholding | 37.10% | 46.45% | 41.25% | 4.60% | $58.9\%$ false-alarm reduction on load shifts |
| **E3** | **Calibrated Fusion (AE + IF)** | **46.31%** | **44.98%** | **45.64%** | **3.05%** | **+5.80 pp F1 gain; 39.60% relative FPR reduction** |
| **E4** | Fusion + Explainability (XAI) | **46.31%** | **44.98%** | **45.64%** | **3.05%** | Adds top-K attribution with zero detection penalty |
| **E5** | **Full System (Fusion + Drift + ZT)** | **46.31%** | **44.98%** | **45.64%** | **3.05%** | Complete end-to-end system with governance |

### 4.2 Zero-Day LOCO (Leave-One-Class-Out) Generalization (Phase 2)

*Evaluated on unseen, held-out attack families without retraining:*

| Model Architecture | Precision | Recall | F1-Score | Zero-Day Detection Delay |
|---|:---:|:---:|:---:|:---:|
| **Baseline Autoencoder** | 24.12% | 18.00% | 20.61% | 14 samples |
| **Residual Autoencoder** | 31.85% | 34.28% | 33.02% | 8 samples |
| **Causal Window Scorer ($L=5$)** | **38.45%** | **35.34%** | **36.83%** | **5 samples** |

### 4.3 Concept Drift 4-Scenario Benchmark (Phase 4)

| Test Scenario | Sample Count | Injected Dynamics | Detector Response | Result Status |
|---|:---:|---|---|:---:|
| **Scenario 1: Stable Normal Stream** | 15,000 | Steady baseline telemetry | 0 alarms raised ($S_k = 0$) | **PASS** |
| **Scenario 2: Sustained Step Shift** | 10,000 | $+1.5\sigma$ mean operational shift | Alarm at sample 7,849; tags `REVIEW_REQUIRED` | **PASS** |
| **Scenario 3: Short Cyber Attack Burst** | 500 | Severe 10-sample attack bursts | Block Page-Hinkley absorbs burst context | **PASS** |
| **Scenario 4: Transient Noise Spike** | 1,000 | 15-sample random Gaussian noise | No false drift triggers | **PASS** |

---

## 5. Resource Efficiency & Edge Hardware Benchmarks (Phase 7)

*Benchmarked on single CPU core (AMD/Intel x86_64, equivalent to ARM Cortex-A72 on Raspberry Pi 4):*

| Pipeline Component | Latency (Median) | Single-Sample Throughput | Batch Throughput ($B=64$) | Peak RAM | Edge 4 GB Limit Check |
|---|:---:|:---:|:---:|:---:|:---:|
| **OT Autoencoder (NumPy)** | **0.0109 ms** | 78,693 samples/s | 2,140,000 samples/s | 237.9 MB | **PASS** (<6% of 4 GB) |
| **IT Realistic Autoencoder** | **0.0135 ms** | 71,114 samples/s | 1,080,000 samples/s | 240.5 MB | **PASS** |
| **OT Calibrated Fusion** | **5.41 ms** | 164.5 samples/s | 8,920 samples/s | 244.6 MB | **PASS** |
| **Full Pipeline (Fusion+Drift+ZT+XAI)** | **6.41 ms** | **146.8 samples/s** | **6,400 samples/s** | **258.9 MB** | **PASS** |

> **Real-Time Margin**: The full ResiNet-IDAM pipeline processes a sample in **6.41 ms**, which is **155 times faster** than the standard 1,000 ms (1 Hz) industrial SCADA cycle time.

---

## 6. Full Regression Test Suite Status

```
=================== Full Regression Test Suite Verification ===================
12 Test Suites Executed:
  - tests/test_phase7_benchmarks.py    (5 passed)
  - tests/test_explainability.py       (5 passed)
  - tests/test_fusion.py               (7 passed)
  - tests/test_drift_monitor.py        (7 passed)
  - tests/test_adaptive_threshold.py   (5 passed)
  - tests/test_phase2_zero_day.py      (5 passed)
  - tests/test_dataset_validation.py   (2 passed)
  - tests/test_dataset_loader.py       (2 passed)
  - tests/test_risk_engine.py          (7 passed, 4 subtests)
  - tests/test_temporal_features.py    (3 passed)
  - tests/test_ids2025_preparation.py  (2 passed)
  - tests/test_detection_service.py    (6 passed)

Summary: 56 passed, 4 subtests passed in 41.62s (0 failures, 0 regressions)
==============================================================================
```

