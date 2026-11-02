# ResiNet-IDAM v2.0 Deployment and Operational Guide

ResiNet-IDAM v2.0 is an integrated IT-OT intrusion detection and Zero-Trust risk assessment framework. It provides dual-pipeline telemetry inspection (enterprise IT network flows and industrial OT physical sensor telemetry), adaptive thresholds with poisoning safeguards, statistical concept drift monitoring, multi-model decision fusion, and explainable SecOps alert triage.

---

## 1. Quick Start: Local Windows / Linux Execution

### Prerequisites
- Python 3.11 – 3.13
- 4 GB RAM minimum (tested on resource-constrained setups)

### Setup
```powershell
# Clone and enter directory
cd D:\resinet-idam\ResiNet-IDAM

# Activate virtual environment
.\.venv-ae\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Run Streamlit SecOps Console (Port 8501)
streamlit run app.py

# Optional: Run FastAPI REST Service (Port 8000)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## 2. Containerized Deployment (Docker & Docker-Compose)

ResiNet-IDAM provides multi-stage, non-root Docker deployment.

### Starting with Docker Compose
```bash
docker compose up --build -d
```

- **SecOps Operations Dashboard**: `http://localhost:8501`
- **REST Telemetry Ingestion API**: `http://localhost:8000/docs`

### Service Architecture in Compose
- `resinet-console`: Runs Streamlit UI on port `8501`.
- `resinet-api`: Runs FastAPI ingestion engine on port `8000`.
- Persistent volumes mounted for `./data`, `./models`, and `./reports`.

---

## 3. Resource Efficiency & Edge IoT Gateway Benchmarks

In accordance with Priority 7 of the Upgrade Plan, all models and pipelines were benchmarked for latency, throughput, memory consumption, and disk footprint under real-time streaming (`batch_size=1`) and micro-batching (`batch_size=64`):

| Domain | Component / Pipeline | Batch Size | Median Latency (ms) | Throughput (samples/s) | Peak RAM (MB) | Model Size (MB) |
|---|---|---|---|---|---|---|
| **OT** | OT Autoencoder (NumPy) | 1 | **0.0109 ms** | **78,693** | 185 MB | 0.022 MB |
| **OT** | OT Isolation Forest | 1 | 5.4982 ms | 169.2 | 215 MB | 1.494 MB |
| **OT** | **Calibrated OT Fusion Pipeline** | 1 | **5.4097 ms** | **164.5** | 220 MB | 1.516 MB |
| **OT** | **Full End-to-End System** (Fusion + Drift + ZT + XAI) | 1 | **6.4124 ms** | **146.8** | 225 MB | 1.516 MB |
| **IT** | Random Forest (Supervised 79 feat) | 1 | 35.89 ms | 27.8 | 240 MB | 27.07 MB |
| **IT** | Residual Autoencoder (NumPy) | 1 | **0.0142 ms** | **68,450** | 190 MB | 0.057 MB |
| **IT** | Isolation Forest | 1 | 5.58 ms | 166.3 | 210 MB | 0.706 MB |
| **IT** | **Calibrated IT Fusion Pipeline** | 1 | 41.51 ms | 23.9 | 260 MB | 27.84 MB |
| **OT** | OT Autoencoder (Micro-batch 64) | 64 | **0.0004 ms** | **> 2,000,000** | 195 MB | 0.022 MB |
| **IT** | Residual Autoencoder (Micro-batch 64) | 64 | **0.0009 ms** | **> 980,000** | 200 MB | 0.057 MB |

### Comparison with ZTA-AIDF Paper Claims
The ZTA-AIDF research paper describes an IoT gateway framework targeting a Raspberry Pi 4 (4 GB RAM). ResiNet-IDAM v2.0 easily satisfies this constraint:
- Peak RAM during continuous streaming inference remains **< 260 MB** (well below the 4 GB hardware ceiling).
- Portable NumPy forward engines require zero heavy GPU or CUDA runtimes, allowing sub-millisecond execution directly on edge gateway CPUs.

---

## 4. Master E0 to E5 Ablation Summary

The controlled ablation study evaluated on identical frozen test splits (`reports/full_ablation_e0_to_e5.csv`):

| Exp | Domain | Configuration | Precision | Recall | $F_1$-Score | FPR | Rel $F_1$ Gain | Rel FPR Reduction |
|---|---|---|---|---|---|---|---|---|
| **E0** | **OT** | Current Baseline (Fixed $\tau$) | 34.92% | 46.39% | 39.84% | 5.05% | Baseline | Baseline |
| **E1** | **OT** | E0 + Adaptive Threshold | 32.54% | 43.17% | 37.11% | 5.23% | -6.85% | -3.56% |
| **E2** | **OT** | E0 + Drift Monitor (Block-PH) | 34.92% | 46.39% | 39.84% | 5.05% | Preserved | Preserved |
| **E3** | **OT** | **E0 + Model Fusion** | **46.31%** | 44.98% | **45.64%** | **3.05%** | **+14.56%** | **+39.60%** |
| **E4** | **OT** | E0 + Explainability ($e_j$) | 34.92% | 46.39% | 39.84% | 5.05% | Preserved | Preserved |
| **E5** | **OT** | Full Combined Framework | 28.92% | 49.18% | 36.42% | 7.06% | -8.58% | -39.80% |
| **E0** | **IT** | Current Baseline (RF 79 feat) | 99.48% | 99.94% | 99.71% | 1.30% | Baseline | Baseline |
| **E3** | **IT** | **E0 + Model Fusion** | **99.51%** | 99.74% | **99.62%** | **1.24%** | -0.09% | **+4.62%** |
| **E5** | **IT** | Full Combined Framework | **99.51%** | 99.74% | **99.62%** | **1.24%** | -0.09% | **+4.62%** |

---

## 5. Security & Safety Boundaries
1. **Human-in-the-Loop Governance**: Concept drift detection transitions model state to `REVIEW_REQUIRED`. Strictly inhibits automatic model retraining or automated OT actuation without explicit human operator approval.
2. **Anti-Poisoning Buffer Safeguards**: Suspicious anomalies and high Zero-Trust risk scores are automatically excluded from threshold adaptation buffers to prevent adversarial threshold inflation.
3. **Rollback Mechanism**: Operational thresholds and model weights can be instantly reverted to frozen baselines through one-click operator actions.
