# ResiNet-IDAM: Model & Dataset Audit Report (Sprint 1)

**Audit Date**: November 18, 2026  
**Auditor**: Antigravity Research Engineering  
**Scope**: Verification of IT/OT model artifacts, preprocessing pipelines, dataset provenance, and feature schemas.

---

## 1. Executive Summary

This audit establishes the baseline for ResiNet-IDAM's research evaluation and simulation pipeline as mandated by **Phase 1 (Sprint 1)** of `simulation_improvement_2.txt`.

All primary artifacts for both IT (IDS2025) and OT (HAI) are verified, active, and mathematically compatible with empirical feature schemas:
- **IT Domain**: Supervised 79-feature Random Forest and unsupervised 79-feature NetFlow Autoencoder are verified with frozen validation threshold $\tau_{\text{IT}} = 0.638311$.
- **OT Domain**: 66-feature physical process Autoencoder and `standard_scaler.joblib` (fitted on normal training data only) are verified with frozen validation threshold $\tau_{\text{OT}} = 0.821954$.
- **Ground-Truth Isolation**: In accordance with Rule 4, all evaluation target labels are confirmed to reside outside detector feature matrices.

---

## 2. IT Artifacts Verification (IDS2025 Domain)

| Artifact Path | Format | Features / Shape | Target / Classes | Status | Verification Check |
|---|---|:---:|---|:---:|---|
| `models/it_realistic/random_forest_all.joblib` | Scikit-learn Joblib (27.7 MB) | 79 input features | 7 classes: `['Botnet ARES', 'Brute Force', 'Dos/DDos', 'Infiltration', 'Normal', 'PortScan', 'Web Attack']` | **VERIFIED** | `n_features_in_ = 79`, multi-class probability output |
| `models/it_realistic/random_forest_no_ports.joblib` | Scikit-learn Joblib (38.8 MB) | 77 input features | 7 classes (excluding port columns to test anti-overfitting) | **VERIFIED** | Compatible with 77-feature configuration |
| `models/it_realistic/preprocessing_all.joblib` | Scikit-learn Pipeline (5.1 KB) | 79 input features | Imputer (`SimpleImputer(median)`) + `RobustScaler` | **VERIFIED** | Pipeline matches 79 ordered NetFlow features |
| `models/it_realistic/label_encoder.joblib` | Scikit-learn LabelEncoder (0.5 KB) | 7 classes | Target label mapping: Normal (4), Attacks (0, 1, 2, 3, 5, 6) | **VERIFIED** | Deterministic class-to-index mapping |
| `models/it_realistic/it_autoencoder_weights.npz` | NumPy NPZ (24.3 KB) | 79 -> 32 -> 16 -> 32 -> 79 | Unsupervised reconstruction error | **VERIFIED** | Verified float32 weights |
| `models/it_realistic/it_residual_autoencoder_weights.npz` | NumPy NPZ (58.7 KB) | 79 -> 32 -> 16 + Skips | Skip-connected Residual AE | **VERIFIED** | Multi-head skip connections verified |
| `models/it_realistic/it_autoencoder_threshold.json` | JSON (0.9 KB) | Scalar | $\tau_{\text{IT}} = 0.6383106708526611$ | **VERIFIED** | 99.0th percentile of normal-only validation set |
| `models/it_realistic/loco_iforest_all_botnet_ares.joblib` | Scikit-learn Joblib (722.5 KB) | 79 input features | Density outlier detection for LOCO Botnet evaluation | **VERIFIED** | Calibrated Isolation Forest |

---

## 3. OT Artifacts Verification (HAI Domain)

| Artifact Path | Format | Features / Shape | Target / Classes | Status | Verification Check |
|---|---|:---:|---|:---:|---|
| `models/autoencoder.keras` | Keras / TensorFlow (95.5 KB) | 66 -> 48 -> 32 -> 48 -> 66 | Unsupervised reconstruction error | **VERIFIED** | 66 physical process inputs |
| `models/ot_autoencoder_weights.npz` | NumPy NPZ (22.9 KB) | 66 -> 48 -> 32 -> 48 -> 66 | Lightweight standalone NumPy engine | **VERIFIED** | Verified identical inference to Keras model |
| `models/autoencoder_threshold.json` | JSON (1.4 KB) | Scalar | $\tau_{\text{OT}} = 0.821953535079956$ | **VERIFIED** | Calibrated on normal-only validation data |
| `models/standard_scaler.joblib` | Scikit-learn Joblib (3.1 KB) | 66 features | StandardScaler fitted on training partition only | **VERIFIED** | `n_features_in_ = 66`, mean/variance verified |
| `models/feature_columns.json` | JSON (1.1 KB) | 66 strings | Strict column ordering (`P1_FCV01D` ... `x1003_24_SUM_OUT`) | **VERIFIED** | Exactly matches HAI sensor registers |
| `models/preprocessing_metadata.json` | JSON (1.8 KB) | Metadata dictionary | Preprocessing pipeline specs & scaling parameters | **VERIFIED** | Preserves dataset checksums and parameters |

---

## 4. Dataset Lineage & Empirical Ground-Truth Partitions

### 4.1 IT Dataset: IDS2025
- **Location**: `data/processed/ids2025_sample.csv` (805 rows, 80 columns: 79 features + 1 label).
- **Class Breakdown**:
  - `Normal`: 200 rows (indices 0..199)
  - `Dos/DDos`: 150 rows (indices 200..349)
  - `PortScan`: 150 rows (indices 350..499)
  - `Brute Force`: 100 rows (indices 500..599)
  - `Botnet ARES`: 100 rows (indices 600..699)
  - `Web Attack`: 80 rows (indices 700..779)
  - `Infiltration`: 25 rows (indices 780..804)
- **High-Volume Preprocessed Partition**: `data/processed/it_realistic/` (`it_train_all.npz`: 4.5 MB, `it_test_all.npz`: 1.2 MB, 18,344 frozen test flows).
- **Leakage Prevention**: Feature columns exclude IP addresses, timestamps, and target labels. Row 243 (containing netflow calculation infinity) is handled and sanitized.

### 4.2 OT Dataset: HAI (Hardware-in-the-Loop Augmented ICS)
- **Location**:
  - Training (Normal only): `data/processed/hai-train-scaled.csv` (203.4 MB).
  - Test (Normal + Attacks): `data/processed/hai-test-scaled.csv` (39.4 MB, 54,000 timesteps).
  - Labeled Ground Truth: `data/processed/hai-test-labeled.csv` (30.8 MB).
- **Physical Attack Episodes**:
  - Attack labels start at row 1503 and span across 2,981 attack time-steps (including verified test rows 1600, 1700, 1750, 1800, 2300).
- **Features**: Exactly 66 physical process tags (valves, pumps, pressure transmitters, flow sensors).

---

## 5. Acceptance Audit Conclusion

1. **Model Compatibility**: All models accept strictly their designated schemas (79 for IT, 66 for OT).
2. **Threshold Integrity**: Both $\tau_{\text{IT}} = 0.638311$ and $\tau_{\text{OT}} = 0.821954$ are frozen and derived exclusively from normal validation data.
3. **Artifact Completeness**: `standard_scaler.joblib` is present and verified. No detector is using invented or fabricated default thresholds.

