# ResiNet-IDAM

## Adaptive Deep Learning and Zero-Trust Framework for Detecting Zero-Day Intrusions in IT-OT Hybrid Critical Infrastructures

## 1. Project Overview

ResiNet-IDAM is a research project focused on developing an adaptive, intelligent security framework for detecting suspicious activities and potential zero-day intrusions in hybrid Information Technology (IT) and Operational Technology (OT) environments.

Modern critical infrastructures depend on interconnected IT systems and industrial OT environments. Traditional security solutions may struggle to identify previously unseen attack patterns, particularly when threats cross the boundary between enterprise networks and industrial control systems.

ResiNet-IDAM aims to address this challenge by combining deep-learning-based anomaly detection, IT and OT telemetry analysis, and Zero Trust security principles in a unified framework.

**Core concept:** Monitor system behavior, detect anomalies, assess risk using contextual security signals, and support informed security responses.

## 2. Problem Statement

IT and OT environments have different operational characteristics, but their increasing interconnection creates additional security risks. Conventional detection mechanisms that depend heavily on known attack signatures may fail to identify unfamiliar or evolving threats.

A unified framework is needed to analyze behavior across IT and OT environments, identify potentially malicious deviations, and incorporate identity, device, and contextual information into security decisions.

## 3. Project Objectives

- Develop anomaly-detection mechanisms using deep learning and machine learning.
- Analyze industrial telemetry to identify deviations from normal OT behavior.
- Extend detection capabilities to IT security telemetry.
- Investigate adaptive detection approaches for previously unseen attack patterns.
- Incorporate Zero Trust principles into risk assessment and access-related decisions.
- Correlate IT and OT security findings to support unified monitoring.
- Develop a monitoring dashboard for presenting anomalies, alerts, and security insights.

## 4. Proposed System Architecture

The proposed framework consists of the following components:

1. **IT Data Collection:** Collect relevant enterprise telemetry, such as authentication events, server logs, and network activity.
2. **OT Data Collection:** Process industrial telemetry from sensors, programmable logic controllers (PLCs), supervisory control and data acquisition (SCADA) systems, or suitable benchmark datasets.
3. **Data Preprocessing:** Clean, transform, normalize, and prepare data for analysis.
4. **Anomaly Detection:** Apply machine-learning and deep-learning models to identify deviations from expected behavior.
5. **Adaptive Detection:** Investigate mechanisms for updating detection behavior as operating conditions and threat patterns change.
6. **Zero Trust Risk Assessment:** Evaluate available identity, device, behavior, and contextual signals to support risk-aware security decisions.
7. **IT-OT Correlation:** Combine relevant findings from IT and OT monitoring to identify potentially related security events.
8. **Monitoring and Response:** Present alerts, risk indicators, and detection results through a unified dashboard and support appropriate response workflows.

### Architecture Flow

IT Telemetry + OT Telemetry → Data Preprocessing → Anomaly Detection → Risk Assessment and IT-OT Correlation → Monitoring and Response

The deployed application uses this flow through a shared detection service and a single Streamlit entry point.

## 5. Technology Stack

The technologies used or planned for the project include:

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | pandas, NumPy |
| Baseline anomaly detection | scikit-learn |
| Deep learning | TensorFlow / Keras |
| Model persistence | joblib and Keras model files |
| OT benchmark dataset | HAI industrial telemetry dataset |
| IT detection | IDS2025 Random Forest adapter |
| Risk assessment and Zero Trust | Five-factor contextual risk engine |
| Unified dashboard | Streamlit operations console |

## 6. Current Implementation Status

### Implemented

- HAI industrial telemetry data preparation.
- Feature selection and preprocessing.
- Feature scaling for model inputs.
- Isolation Forest baseline training and evaluation.
- Autoencoder training using a chronological training and validation split.
- Preliminary evaluation against labeled HAI test data.
- Experimental comparison of anomaly-detection performance.
- IDS2025 IT detector adapter with explicit artifact availability checks.
- Unified IT/OT event contract and detection orchestration service.
- Five-factor contextual Zero-Trust risk scoring.
- SQLite alert persistence and review-status lifecycle.
- Streamlit operations, detection, alert, risk, health, and administration screens.
- Optional FastAPI adapter using the same application service.

## 7. Deployment documentation

- **[Architecture](docs/architecture.md)** — application boundaries and runtime flow.
- **[Deployment](docs/deployment.md)** — local setup, artifacts, and startup.
- **[API Contract](docs/api-contract.md)** — optional REST adapter contract.
- **[Dataset Report](docs/dataset-report.md)** — HAI data contract and validation results.
- **[Model Evaluation](docs/model-evaluation.md)** — detector metrics and threshold policy.
- **[Risk Policy](docs/risk-policy.md)** — contextual risk formula and severity actions.

## 8. Dataset Audit

Run the Phase 1 audit before training or evaluating a model:

```powershell
python src/data/inspect_dataset.py
python src/data/validate_dataset.py
```

The commands write machine-readable reports to `reports/`. The validation report explicitly records whether the
training file, scaler artifact, feature ordering, timestamps, labels, and finite-value checks are available and valid.
Training must not proceed until the normal-only training assumption and scaler provenance are verified.

## 9. Local application

The production user interface is a single Streamlit application:

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501`. The application provides operations overview,
detection workspace, alert management, risk assessment, model/data health, and
administration. The optional FastAPI adapter can be started separately for
machine-to-machine integrations.

## 10. Evaluation and limitations

The current OT experiments use the HAI industrial telemetry dataset to investigate anomaly detection with Isolation Forest and an autoencoder.

Evaluation includes attack precision, attack recall, attack F1-score, false positives, and false negatives.

The current results are preliminary and specific to the implemented experiments. They do not independently establish reliable zero-day detection or validate the complete IT-OT and Zero Trust framework.

## 10. Project Limitations

1. **Results Remain Dataset-Specific**: Benchmark performance evaluated on IDS2025 and HAI datasets reflects specific experimental configurations and may not directly represent varied industrial environments.
2. **Synthetic Held-Out-Class Evaluation**: Held-out-class (LOCO) evaluations are synthetic research approximations and do not constitute proof of real-world zero-day detection. The project is a *synthetic held-out-class evaluation and anomaly-based research prototype for novel intrusion detection*, not a guaranteed zero-day defense.
3. **No Genuine IT/OT Time-Synchronization**: IT network flow captures and OT industrial physical telemetry were collected from distinct experimental testbeds and are synthetically correlated across cross-domain attack scenarios.
4. **Decision-Support Boundary**: The framework is strictly a decision-support prototype for security analysts. The contextual Zero-Trust engine recommends triage actions (`ALLOW_AND_MONITOR`, `VERIFY_ADDITIONAL_FACTORS`, `RESTRICT_AND_ALERT`) and **never authorizes automatic physical or industrial control actions**.

## 11. Repository Structure

```text
ResiNet-IDAM/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # Original datasets, obtained separately
│   └── processed/        # Generated/prepared datasets
├── models/               # Locally generated model files
└── src/
    ├── application/       # Shared detection orchestration
    ├── detectors/         # IT and OT detector adapters
    ├── domain/            # Event and detector contracts
    ├── infrastructure/   # Configuration and artifact access
    ├── ui/                # Streamlit presentation package
    └── zero_trust/        # Contextual risk engine
```

Large datasets, trained model binaries, and local databases are deployment artifacts and are not stored in Git.

## 12. Operational boundary

The system is a security-monitoring decision-support application. Identity,
device, and asset context are currently supplied as normalized deployment
inputs; the risk engine does not authorize automatic industrial actions.
