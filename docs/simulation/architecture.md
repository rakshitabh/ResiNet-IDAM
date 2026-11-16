# ResiNet-IDAM Simulation Architecture & Repository Audit Contract

**Document Version:** 1.0.0  
**Status:** Frozen Baseline  
**Scope:** CSV-Driven IT/OT Zero-Trust Event Simulation Architecture Audit (Phase 1)

---

## 1. Repository Audit Findings & Verified Contracts

### 1.1 OT Physical Telemetry Detector (`src/detectors/hai_autoencoder.py`)
- **Model Implementations**: `NumpyAutoencoder` (NumPy forward-pass) and Keras Autoencoder fallback.
- **Input Feature Count**: **66 ordered physical features** (HAI 22.04 standard).
- **Feature Order Contract**: Defined in `models/autoencoder_threshold.json` (`feature_columns`).
- **Constant Features Preserved**: 7 operational features (`P1_PP04`, `P1_SOL01D`, `P1_SOL03D`, `P2_AutoSD`, `P2_Emerg`, `P2_TripEx`, `P3_LH01`).
- **Threshold**: Static threshold $T_{\text{static}} = 0.089945$.
- **Explainability Residual Output**: $e_j = (x_j - \hat{x}_j)^2$, exported as `top_contributing_features` with `feature_name`, `observed_value`, `expected_value`, `residual_error`, and `relative_deviation`.

### 1.2 IT Network Flow Detector (`src/detectors/ids2025.py`)
- **Model Implementations**: `Ids2025Detector` (Random Forest ensemble) and `NumpyItAutoencoder`.
- **Input Feature Count**: **79 flow features** (default `all_features` config) and 77 features (`no_ports` config).
- **Feature Order Contract**: Sourced from `data/processed/it_realistic/metadata.json` (`configurations.all_features.features`).
- **Threshold**: $T_{\text{IT}} = 0.638311$.
- **Preprocessing**: `models/it_realistic/preprocessing_all.joblib`.

### 1.3 Detection Orchestrator (`src/application/detection_service.py`)
- **Routing**: Routes `source="OT"` to `HaiAutoencoderDetector` and `source="IT"` to `Ids2025Detector`.
- **Drift Governance**: Stateful `DomainDriftMonitor` instances for OT (Block Page-Hinkley) and IT (ADWIN).
- **Security Event Creation**: Converts detection output into unified `SecurityEvent` schema with full metadata.

### 1.4 Zero-Trust Risk Engine (`src/zero_trust/risk_engine.py`)
- **Inputs**: `RiskInput` dataclass with 5 normalized factors:
  - `anomaly_severity` ($c_{\text{anom}} \in [0, 1]$): Normalized model anomaly signal.
  - `behavior_deviation` ($c_{\text{beh}} \in [0, 1]$): Contextual operational deviation.
  - `identity_trust` ($\tau_I \in [0, 1]$): Identity validity score ($1 - \tau_I$ = untrust).
  - `device_trust` ($\tau_D \in [0, 1]$): Device health and posture ($1 - \tau_D$ = untrust).
  - `asset_criticality` ($c_{\text{crit}} \in [0, 1]$): Asset impact factor.
- **Verified Risk Formula**:
  $$\text{Risk Score} = 100 \times \left[0.25 c_{\text{anom}} + 0.30 c_{\text{beh}} + 0.15(1 - \tau_I) + 0.10(1 - \tau_D) + 0.20 c_{\text{crit}}\right]$$
- **Severity Bands**:
  - Below 30: **LOW** (`RECORD_AND_MONITOR`)
  - 30 to 59: **MEDIUM** (`ALERT_FOR_REVIEW`)
  - 60 to 84: **HIGH** (`PRIORITIZE_INVESTIGATION`)
  - 85 and above: **CRITICAL** (`RECOMMEND_CONTAINMENT`)
- **Authoritative Policy Decisions**:
  - `evaluate_zero_trust_decision(event)` emits transparent policy tuples:
    - Verdict: `ALLOW`, `REQUIRE VERIFICATION`, `DENY`
    - Action: `RECORD_AND_MONITOR`, `STEP_UP_AUTHENTICATION`, `REVOKE_AND_ISOLATE`, `RECOMMEND_CONTAINMENT`
    - Safety Guarantee: `is_automatic_action = False` (Rule 10 strictly maintained).

---

## 2. Canonical CSV Contract & Upload Protection

### 2.1 15-Column Canonical Schema
```csv
event_id,offset_seconds,source,event_type,user_id,device_id,asset_id,ip_address,mac_address,telemetry_record_id,identity_trust,device_trust,asset_criticality,behavior_deviation,expected_outcome
```

### 2.2 Upload Validation Limits
- **Maximum File Size**: 10 MB (10,485,760 bytes).
- **Maximum Event Rows**: 10,000 scheduled rows.
- **Maximum Offset Seconds**: 86,400.0 seconds (24 hours).
- **Strict Checks**:
  1. Header matches canonical columns or permitted alias subset.
  2. `event_id` non-empty and unique across rows.
  3. `offset_seconds` non-negative float.
  4. `source` strictly in `{"IT", "OT"}`.
  5. Trust & criticality values in $[0.0, 1.0]$.
  6. `expected_outcome` optional, restricted to `{"ALLOW", "REQUIRE_VERIFICATION", "DENY"}`.

---

## 3. Telemetry Resolution Architecture

To eliminate the "synthetic data fallacy", the simulator enforces deterministic telemetry resolution:
1. `telemetry_record_id` is an index, timestamp, or row key into genuine datasets:
   - OT: `data/processed/hai-test-labeled.csv` or `data/processed/hai-test-scaled.csv` (rows 0 to 54,000).
   - IT: `data/processed/ids2025_sample.csv` (rows 0 to 800).
2. The `TelemetryResolver` pre-loads and caches these indexed records.
3. During event execution, the resolver fetches the exact 66-dim (OT) or 79-dim (IT) feature vector and feeds it to `detection_service.detect`.
4. If `telemetry_record_id` is omitted for an event, the resolver pulls a deterministic baseline record corresponding to `event_type` and logs it explicitly in the execution trace.

---

## 4. Simulation Engine & State Machine

### 4.1 State Definitions
- **Run States**: `READY`, `RUNNING`, `PAUSED`, `STOPPING`, `STOPPED`, `COMPLETED`, `FAILED`.
- **Event States**: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`.

### 4.2 Monotonic Clock Scheduling
- The simulation clock $t_{\text{sim}}$ runs monotonically:
  $$\Delta t_{\text{sim}} = \Delta t_{\text{wall}} \times \text{speed\_multiplier}$$
- Supported multipliers: $1\times, 2\times, 5\times, 10\times$.
- Pause freezes $t_{\text{sim}}$; Resume continues without time drift.
- Stop reconciles in-flight work with a bounded timeout (500 ms) and marks pending events as `CANCELLED`.
- Restart creates a fresh `run_id` and archives the previous run to `data/simulation/runs/<run_id>.json`.

---

## 5. Storage Isolation Strategy

- Normal application alerts remain untouched in `data/resinet-events.sqlite3`.
- Simulation runs are stored in:
  - `data/simulation/runs/<run_id>.json`: Complete machine-readable execution trace.
  - `data/simulation/reports/<run_id>.csv`: Event-level tabular results.
  - `data/simulation/reports/<run_id>.json`: Run-level and event-level summary for UI downloads.
  - `data/simulation/uploads/`: Sanitized uploaded user CSV files.

