# ResiNet-IDAM
## Master Project Blueprint
### Product Requirements · Software Requirements · System Design · Implementation Roadmap

**Document version:** 1.0  
**Project type:** Academic cybersecurity research prototype  
**Deployment:** Local laptop demonstration  
**Primary dataset:** HAI — HIL-based Augmented ICS dataset  
**Development approach:** Phased, modular implementation  
**Status:** Initial baseline specification

---

# 1. Executive Summary

## 1.1 Project Overview

ResiNet-IDAM is a research prototype designed to detect anomalous industrial process behavior using deep learning and prioritize security events using a simulated Zero-Trust risk assessment framework.

The system analyzes industrial telemetry from the HAI dataset, detects deviations from learned process behavior, combines anomaly severity with simulated identity and device trust information, and displays security events through a local web dashboard.

The prototype is intended to demonstrate the integration of industrial anomaly detection, contextual risk scoring and security monitoring in an IT–OT cybersecurity setting.

## 1.2 Problem Statement

Industrial Control Systems (ICS) rely on sensors, controllers, actuators and interconnected operational technology to manage physical processes. Unexpected or malicious changes in process behavior can affect availability, reliability and safety.

Conventional detection approaches may struggle with previously unseen patterns or may generate alerts without sufficient contextual information to prioritize investigation.

ResiNet-IDAM addresses this research problem through three integrated capabilities:

1. Deep-learning-based anomaly detection.
2. Context-aware risk scoring using simulated Zero-Trust identity and device attributes.
3. A centralized dashboard for event inspection, prioritization and evaluation.

## 1.3 Project Objectives

- Prepare and validate industrial time-series telemetry.
- Train an anomaly detector primarily on verified normal process behavior.
- Identify unusual observations using reconstruction error and a validation-selected threshold.
- Evaluate detection performance against ground-truth labels.
- Simulate identity trust, device trust and asset criticality.
- Calculate an interpretable risk score between 0 and 100.
- Display anomaly events, risk levels and event details in a local dashboard.
- Maintain reproducible experiments, tests and model artifacts.

## 1.4 Scope Boundaries

### In scope

- Offline analysis of the HAI dataset.
- Dense Autoencoder baseline.
- Model evaluation and error analysis.
- Simulated identity and device context.
- Deterministic risk-scoring rules.
- Local API, dashboard and event history.
- Optional experimental comparison with additional models.

### Out of scope for the initial version

- Production deployment in a real industrial environment.
- Direct control of pumps, valves, turbines or other physical equipment.
- Real identity-provider integration.
- Guaranteed detection of every zero-day exploit.
- Mandatory Kafka, MQTT or distributed infrastructure.
- Automatic model retraining without validation and approval.

---

# 2. Product Requirements Document (PRD)

## 2.1 Target Users

**Security analyst:** Reviews anomalies and prioritizes suspicious events.

**System administrator:** Inspects model status, configuration and application health.

**Project evaluator:** Reviews the architecture, methodology, evaluation metrics and demonstration.

**Development team:** Implements, tests and maintains individual modules according to the shared specification.

## 2.2 Product Workflow

1. Load the prepared HAI training and test datasets.
2. Validate feature names, types, row counts and timestamps.
3. Train or load the anomaly detection model.
4. Calculate reconstruction errors for incoming or replayed observations.
5. Apply the configured threshold to flag anomalies.
6. Associate observations with simulated identity, device and asset context.
7. Calculate a contextual risk score.
8. Store the resulting event and its score breakdown.
9. Display alerts and supporting details in the dashboard.
10. Evaluate offline predictions against ground-truth labels.

## 2.3 Core Features

| ID | Feature | Priority |
|---|---|---|
| PRD-01 | Dataset validation and preprocessing | Must have |
| PRD-02 | Autoencoder training and inference | Must have |
| PRD-03 | Anomaly threshold configuration | Must have |
| PRD-04 | Model evaluation and reporting | Must have |
| PRD-05 | Simulated identity/device trust | Must have |
| PRD-06 | Risk score and severity classification | Must have |
| PRD-07 | Security event API | Must have |
| PRD-08 | Dashboard overview and event inspection | Must have |
| PRD-09 | Persistent event history | Must have |
| PRD-10 | Alternative model comparison | Should have |
| PRD-11 | Controlled model retraining | Could have |
| PRD-12 | BiLSTM and streaming integrations | Could have |

## 2.4 Success Criteria

The project is considered functionally successful when:

- The dataset preparation and model training processes are reproducible.
- The saved model can be loaded independently of training.
- The model generates an anomaly score for each valid input observation.
- Evaluation metrics are computed using held-out ground-truth labels.
- The risk engine generates a valid, explainable score from 0 to 100.
- The dashboard displays event details and risk classifications.
- The complete workflow can be demonstrated locally.
- Results, limitations and known failure cases are documented.

Detection quality must be reported using measured results; no minimum accuracy or detection rate is claimed before experimentation.

---

# 3. Software Requirements Specification (SRS)

## 3.1 Functional Requirements

### FR-01: Dataset Ingestion

The system shall load the prepared training and test CSV files from configured paths.

It shall report missing files, invalid columns, non-numeric features and invalid label values.

### FR-02: Feature Validation

The system shall maintain an explicit ordered list of model features.

Timestamp and ground-truth label columns shall not be included in the model input.

Training and inference features shall have identical names and ordering.

### FR-03: Model Training

The system shall train an Autoencoder using an approved training dataset.

The team shall verify whether training data contains only normal samples before using it for normal-only training.

The training process shall support reproducible configuration and model artifact saving.

### FR-04: Anomaly Detection

For each observation, the system shall calculate a reconstruction error.

The system shall classify an observation as anomalous when its error exceeds the configured threshold.

The threshold shall be selected using appropriate validation data, not tuned on the final test set.

### FR-05: Model Evaluation

The system shall calculate:

- Confusion matrix.
- Precision.
- Recall.
- F1-score.
- False-positive rate.
- PR-AUC and ROC-AUC where applicable.

The evaluation report shall record the model version, feature configuration and threshold.

### FR-06: Simulated Zero-Trust Context

The system shall associate events with simulated identities, devices and industrial assets.

Context shall include identity trust, device trust and asset criticality.

The documentation shall explicitly identify these attributes as simulated rather than real authentication evidence.

### FR-07: Risk Assessment

The system shall calculate a contextual risk score from 0 to 100.

It shall retain the individual scoring components and resulting severity classification for inspection.

### FR-08: Event Management

The system shall record the timestamp, asset, anomaly score, anomaly decision, risk score, severity and event status.

Ground-truth labels shall be retained only for offline evaluation and shall not influence inference decisions.

### FR-09: Dashboard

The dashboard shall display:

- Total events processed.
- Detected anomalies.
- Risk distribution.
- Event timestamps.
- Affected assets.
- Risk levels.
- Individual event details.
- Model evaluation information.

### FR-10: Error Handling

The system shall handle missing input files, malformed requests, incompatible feature schemas and unavailable model artifacts without silently producing invalid predictions.

## 3.2 Non-Functional Requirements

**NFR-01 — Reproducibility:** Training configuration, feature selection and threshold must be recorded.

**NFR-02 — Modularity:** Data preparation, detection, risk scoring, API and frontend must be separate modules.

**NFR-03 — Maintainability:** Shared schemas and configuration must be documented and version-controlled.

**NFR-04 — Performance:** The application must process the prepared dataset on the available development laptop. Actual throughput and latency must be measured.

**NFR-05 — Security:** Validate API inputs, restrict local services to the intended environment and avoid exposing sensitive configuration.

**NFR-06 — Reliability:** Missing models or invalid features must generate explicit errors rather than fabricated alerts.

**NFR-07 — Explainability:** An event must expose its anomaly score, threshold, risk components and final classification.

**NFR-08 — Safety:** The prototype must not issue commands to real industrial control equipment.

---

# 4. System Design

## 4.1 High-Level Architecture

The system shall follow a modular, layered architecture.

**Layer 1 — Data Layer**
- Original HAI files.
- Prepared training and test CSVs.
- Feature configuration.
- Evaluation labels.
- Persisted security events.

**Layer 2 — ML Layer**
- Feature validation.
- Autoencoder training.
- Reconstruction error calculation.
- Threshold-based anomaly detection.
- Evaluation and model artifact management.

**Layer 3 — Risk Intelligence Layer**
- Simulated identity context.
- Simulated device context.
- Asset criticality.
- Risk calculation and severity classification.

**Layer 4 — Application Layer**
- FastAPI endpoints.
- Event persistence.
- Model status and metrics.
- Input validation and error handling.

**Layer 5 — Presentation Layer**
- React dashboard.
- Event table.
- Risk summaries.
- Event detail view.
- Evaluation metrics.

## 4.2 Component Interaction

HAI CSV → Data validation → Feature matrix → Autoencoder → Reconstruction error → Anomaly decision → Simulated context → Risk engine → Event storage → FastAPI → React dashboard.

Offline evaluation follows a separate path: predicted anomaly scores are compared with held-out ground-truth labels to calculate metrics.

The ground-truth labels must not be passed into the inference or risk-scoring pipeline.

## 4.3 Data Flow

### Training flow

1. Load training data.
2. Validate the feature schema.
3. Confirm normal-only training eligibility.
4. Create a chronological training/validation split when appropriate.
5. Train the Autoencoder.
6. Calculate validation reconstruction errors.
7. Select and document the threshold.
8. Save the model, feature list and threshold.

### Evaluation flow

1. Load the saved model and threshold.
2. Load test features without the label column.
3. Calculate reconstruction errors.
4. Generate anomaly predictions.
5. Compare predictions with ground-truth labels in the evaluation module.
6. Generate a metrics report and timestamped score file.

### Risk flow

1. Receive an anomaly score and decision.
2. Resolve the simulated identity, device and asset context.
3. Normalize the scoring inputs.
4. Calculate the risk score.
5. Assign a severity band.
6. Store the event and its score breakdown.

---

# 5. Dataset Specification

## 5.1 Dataset Source

Official repository: https://github.com/icsdataset/hai

The project uses the HAI industrial-control-system dataset as its process telemetry source.

## 5.2 Prepared Dataset Inventory

The current prepared files are expected to contain:

| File | Expected structure |
|---|---|
| `hai-train-scaled.csv` | 280,800 rows, timestamp and 66 process features |
| `hai-test-scaled.csv` | 54,000 rows, timestamp, 66 process features and label |
| `standard_scaler.joblib` | Fitted preprocessing object, if produced by the existing pipeline |

The team must verify these dimensions and the source dataset version against the actual files.

The training CSV does not contain labels. Therefore, normal-only training must be verified from the original dataset documentation or preparation procedure before training begins.

## 5.3 Data Integrity Rules

- Preserve original raw files unchanged.
- Keep raw and processed data separate.
- Exclude timestamps and labels from the model feature matrix.
- Apply the training-fitted scaler to test features.
- Keep feature ordering consistent.
- Document removed features and the reasons for removing them.
- Investigate features that are constant during training but vary during testing.
- Verify timestamp alignment and missing values.
- Do not tune the final threshold on test labels.

## 5.4 Model Evaluation Protocol

The model shall be evaluated on a held-out test set after preprocessing and threshold selection are finalized.

The evaluation shall include precision, recall, F1-score, false-positive rate and confusion matrix. PR-AUC is especially relevant where attack events are uncommon.

Time-series-aware metrics may be added where the team can implement and validate them correctly.

The report shall disclose limitations arising from dataset imbalance, temporal dependencies and differences between benchmark attacks and real-world zero-day attacks.

---

# 6. Machine Learning Design

## 6.1 Baseline Model

The initial model shall be a dense Autoencoder implemented using TensorFlow/Keras.

The model learns to reconstruct input feature vectors. Its reconstruction error is used as an anomaly score.

A high reconstruction error indicates that the observation differs from the patterns the model learned. It does not independently prove malicious activity.

## 6.2 Threshold Selection

The first baseline shall select its threshold using verified normal validation observations.

A high percentile of normal validation errors may be used as an initial threshold-selection method. The percentile shall be documented and evaluated for false-positive behavior.

The threshold must not be selected by maximizing performance on the final test set.

## 6.3 Artifact Management

The model module shall save:

- Trained Autoencoder.
- Ordered feature list.
- Preprocessing configuration.
- Selected anomaly threshold.
- Training configuration.
- Evaluation metrics.
- Model version and creation metadata.

## 6.4 Optional Experiments

Only after the baseline works end to end may the team evaluate:

- Isolation Forest as a baseline comparison.
- BiLSTM or another temporal model.
- Controlled retraining with validation and rollback.
- Streaming event replay.

Any alternative model must be evaluated using the same documented data split and metrics.

---

# 7. Zero-Trust Risk Engine

## 7.1 Purpose

The risk engine combines model output with contextual security information. It is a separate module from the anomaly detector.

Because the current dataset does not provide a complete identity and device-trust system, the prototype shall use explicitly simulated context.

## 7.2 Context Schema

Each event may reference:

- `identity_id`
- `device_id`
- `asset_id`
- `identity_trust`
- `device_trust`
- `asset_criticality`
- `behavior_deviation`
- `anomaly_severity`

All normalized inputs must lie between 0 and 1.

Higher trust values mean greater confidence in the simulated identity or device. Higher criticality means the asset is more important to protect.

## 7.3 Initial Risk Formula

Define:

- A = anomaly severity.
- B = behavior deviation.
- I = identity trust.
- D = device trust.
- C = asset criticality.

The initial configurable risk formula is:

Risk = 100 × [0.40A + 0.20B + 0.20(1 − I) + 0.10(1 − D) + 0.10C]

This is a proposed engineering baseline, not an experimentally validated security standard. The team must document the weights and evaluate their effect.

## 7.4 Risk Bands

| Score | Classification |
|---|---|
| 0–29 | Low |
| 30–59 | Medium |
| 60–79 | High |
| 80–100 | Critical |

These bands are configurable.

## 7.5 Response Policy

- **Low:** Record and monitor.
- **Medium:** Raise an alert for review.
- **High:** Prioritize investigation and additional verification.
- **Critical:** Recommend containment for a human operator to assess and approve.

The prototype shall not automatically control physical industrial processes.

---

# 8. API Design

The backend shall expose versioned, documented endpoints under `/api`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Application health |
| GET | `/api/summary` | Event and risk summaries |
| GET | `/api/events` | Filtered event listing |
| GET | `/api/events/{event_id}` | Individual event details |
| GET | `/api/model/metrics` | Model evaluation metrics |
| GET | `/api/model/info` | Model version and threshold |
| POST | `/api/events/score` | Score a valid observation using the loaded model and risk engine |

The scoring endpoint shall validate the feature schema and reject missing, extra or incompatible features according to the documented contract.

The inference endpoint shall not accept ground-truth labels as inputs.

The API contract, response schema and error format must be agreed upon before frontend integration.

---

# 9. Frontend Design

## 9.1 Dashboard Overview

Display:
- Total observations processed.
- Number of anomalies detected.
- Distribution of risk levels.
- Recent alerts.
- Model version and status.

## 9.2 Event Table

Columns:
- Timestamp.
- Asset identifier.
- Anomaly score.
- Anomaly decision.
- Risk score.
- Severity.
- Review status.

Support filtering by time range, severity, asset and review status.

## 9.3 Event Detail

Display:
- Event metadata.
- Anomaly score and threshold.
- Simulated identity and device context.
- Risk formula components.
- Final score and severity.
- Ground-truth label only in an explicitly marked offline evaluation view.

## 9.4 Evaluation Page

Display:
- Precision.
- Recall.
- F1-score.
- False-positive rate.
- Confusion matrix.
- PR-AUC and ROC-AUC where available.

The interface must distinguish measured model performance from simulated security context.

---

# 10. Technology Stack

| Component | Selected technology |
|---|---|
| Language | Python 3.10+ |
| Data processing | pandas, NumPy |
| Preprocessing and metrics | scikit-learn |
| Deep learning | TensorFlow/Keras |
| Model persistence | Keras model format, Joblib |
| Backend | FastAPI |
| ASGI server | Uvicorn |
| Frontend | React + Vite |
| Charts | Recharts |
| Local database | SQLite |
| Testing | pytest |
| Version control | Git + GitHub |
| Optional deployment packaging | Docker Compose |

The team shall use one primary deep-learning framework. Additional infrastructure must be justified by an approved requirement rather than introduced pre-emptively.

Exact package versions shall be pinned in the project's dependency files after the development environment is verified.

---

# 11. Implementation Roadmap

## Phase 0 — Planning and Repository Setup

**Activities**
- Confirm scope and project terminology.
- Agree on repository structure and coding conventions.
- Assign module owners.
- Record environment and dependency versions.
- Define the shared API and data contracts.

**Deliverables**
- Approved master blueprint.
- Repository structure.
- Initial README.
- Dependency configuration.
- Team responsibility matrix.

**Exit criteria:** All teammates agree on the architecture, interfaces and boundaries.

## Phase 1 — Dataset Understanding and Validation

**Activities**
- Inspect source files and prepared CSVs.
- Verify columns, row counts, data types and timestamps.
- Check label alignment and class distribution.
- Confirm normal-only training eligibility.
- Verify scaling and feature selection.
- Document data limitations.

**Deliverables**
- Dataset report.
- Feature schema.
- Validated processed data.
- Reproducible preparation instructions.

**Exit criteria:** Dataset assumptions are verified and recorded.

## Phase 2 — Baseline Anomaly Detection

**Activities**
- Implement the Autoencoder.
- Define training and validation splits.
- Select the threshold using validation data.
- Save model artifacts.
- Generate anomaly scores.
- Evaluate against held-out test labels.

**Deliverables**
- Training and evaluation scripts.
- Saved model and threshold.
- Metrics report.
- Timestamped anomaly-score output.

**Exit criteria:** The model loads independently, produces reproducible scores and has a documented evaluation report.

## Phase 3 — Zero-Trust Risk Engine

**Activities**
- Define identity, device and asset schemas.
- Implement deterministic simulated context.
- Implement risk formula and severity bands.
- Add unit tests for boundaries and invalid inputs.

**Deliverables**
- Risk-scoring module.
- Context configuration.
- Risk policy documentation.
- Unit tests.

**Exit criteria:** Identical valid inputs produce identical scores, and every score has an inspectable breakdown.

## Phase 4 — Backend and Persistence

**Activities**
- Implement FastAPI endpoints.
- Load the saved model and configuration.
- Validate incoming feature data.
- Integrate risk scoring.
- Persist and retrieve event records.
- Add API error handling and tests.

**Deliverables**
- Running local API.
- Documented API schema.
- Event storage.
- Integration tests.

**Exit criteria:** Valid observations produce consistent anomaly and risk results through the API.

## Phase 5 — Security Dashboard

**Activities**
- Build overview and event views.
- Integrate frontend with the API.
- Implement filters and event details.
- Add evaluation charts.
- Handle loading, empty and error states.

**Deliverables**
- Functional React dashboard.
- Integrated event views.
- Evaluation page.

**Exit criteria:** A reviewer can trace a sample event from its anomaly score to its risk classification.

## Phase 6 — Integration, Testing and Demonstration

**Activities**
- Run the complete local workflow.
- Verify data and model reproducibility.
- Test error handling and edge cases.
- Capture model metrics and screenshots.
- Prepare the presentation and final report.
- Document limitations and future work.

**Deliverables**
- Working local demonstration.
- Test report.
- Final technical documentation.
- Demonstration script and presentation.

**Exit criteria:** All must-have requirements are implemented, tested and demonstrated.

## Phase 7 — Optional Research Extensions

Only after the core project passes its acceptance criteria:

- Compare against Isolation Forest.
- Investigate temporal models.
- Implement controlled retraining.
- Add streaming replay or optional messaging infrastructure.

**Exit criteria:** Each extension demonstrates measurable value and does not destabilize the baseline.

---

# 12. Team Responsibilities

Each module shall have one accountable owner. Contributors may assist across modules, but the owner is responsible for integration and acceptance.

| Workstream | Responsibilities | Required outputs |
|---|---|---|
| Data and ML | Data validation, training, threshold selection and evaluation | Dataset report, model and metrics |
| Zero-Trust | Context schema, risk formula and policies | Risk engine and tests |
| Backend | API contracts, inference integration and persistence | FastAPI service and API tests |
| Frontend | Dashboard, event inspection and charts | React application |
| Integration and QA | Cross-module testing, documentation and demonstration | Test report and final demo |

For a smaller team, members may own multiple workstreams. Responsibility assignments shall be recorded in the repository.

## Shared Rules

1. No teammate may silently change the feature schema.
2. Changes to the API contract must be communicated before integration.
3. Model thresholds and risk weights must be configuration-driven.
4. Each module must have a clear input/output contract.
5. Every feature must include an acceptance criterion and relevant tests.
6. Experimental metrics must be reproducible and honestly reported.
7. Changes must be committed through the agreed Git workflow.
8. Integration must happen incrementally rather than at the end of the project.

---

# 13. Repository Structure

```text
ResiNet-IDAM/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── config/
├── src/
│   ├── data/
│   ├── detection/
│   ├── risk/
│   ├── api/
│   └── common/
├── frontend/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── acceptance/
├── docs/
│   ├── master-blueprint.md
│   ├── dataset-report.md
│   ├── model-evaluation.md
│   └── api-contract.md
├── requirements.txt
├── README.md
└── .gitignore
```

This is the target structure. Existing scripts and directories should be reused where practical rather than duplicated.

Large datasets and generated model artifacts should not be committed to Git by default. Document their source, storage location and reproduction procedure.

---

# 14. Testing and Acceptance Strategy

## 14.1 Unit Testing

Test:
- Dataset schema validation.
- Feature ordering.
- Model input validation.
- Threshold decisions.
- Risk-score boundaries.
- Severity classification.
- Invalid and missing values.

## 14.2 Integration Testing

Test:
- API loading of the saved model.
- Anomaly inference and risk calculation.
- Event persistence and retrieval.
- API responses consumed by the frontend.
- Correct handling of unavailable artifacts.

## 14.3 Acceptance Testing

The complete system shall demonstrate the following:

1. Load validated HAI data.
2. Score a held-out observation.
3. Produce an anomaly decision.
4. Associate simulated security context.
5. Calculate a risk score.
6. Store the event.
7. Display the event in the dashboard.
8. Show model evaluation results separately from inference results.

## 14.4 Performance Testing

Measure:
- Training duration.
- Inference latency.
- Batch processing throughput.
- Memory consumption.
- Dashboard loading time.

Record actual results on the development laptop. Do not claim performance targets have been met until they are measured.

---

# 15. Git and Collaboration Workflow

## Branching

Use:
- `main` for stable, reviewed code.
- `feature/data-pipeline`
- `feature/anomaly-detection`
- `feature/risk-engine`
- `feature/backend-api`
- `feature/security-dashboard`
- `feature/integration-tests`

Adapt branch names to the team's existing repository conventions.

## Contribution Process

1. Create a feature branch.
2. Implement one clearly defined task.
3. Add tests and update relevant documentation.
4. Run the required checks.
5. Open a pull request with a concise description and test results.
6. Review the change for interface compatibility.
7. Merge only after required checks pass.

## Shared Contracts

Before parallel implementation begins, agree on:

- Ordered feature schema.
- Model artifact paths and metadata.
- Anomaly result schema.
- Risk context and result schema.
- API endpoint definitions.
- Event persistence format.
- Error response format.

These contracts are the main mechanism for preventing incompatible modules.

---

# 16. Risk Register

| Risk | Mitigation |
|---|---|
| Training data includes attacks | Verify source data and filter using trusted information where available |
| Test data influences threshold selection | Freeze threshold selection before final evaluation |
| Feature mismatch between train and test | Maintain a versioned ordered feature list |
| Constant training features change in test | Investigate feature behavior and document selection decisions |
| False-positive volume is high | Analyze normal validation errors and adjust the threshold using validation data |
| Imbalanced test labels distort accuracy | Report precision, recall, F1 and false-positive rate |
| Simulated trust is mistaken for real authentication | Label simulated attributes explicitly throughout the system |
| Model performance does not generalize to real attacks | State benchmark limitations and avoid universal zero-day claims |
| Team integration fails late | Define schemas early and integrate continuously |
| Scope grows excessively | Defer advanced models and streaming until core acceptance criteria pass |

---

# 17. Final Demonstration Plan

The final demonstration should follow one coherent event through the system.

**Step 1:** Show the HAI dataset and explain timestamps, process features and labels.

**Step 2:** Show the trained Autoencoder, threshold and measured evaluation metrics.

**Step 3:** Replay a held-out observation through the inference pipeline.

**Step 4:** Show its reconstruction error and anomaly decision.

**Step 5:** Associate simulated identity, device and asset context.

**Step 6:** Display the resulting risk score and component breakdown.

**Step 7:** Open the dashboard and inspect the event, timestamp and severity.

**Step 8:** Show the evaluation report and explain false positives, missed attacks and limitations.

Use known test records for reproducible demonstrations. Clearly distinguish an observed benchmark attack label from a model prediction and from the simulated Zero-Trust assessment.

---

# 18. Final Project Deliverables

The team shall deliver:

1. Master project blueprint.
2. Software requirements specification.
3. Dataset validation report.
4. Reproducible preprocessing pipeline.
5. Trained Autoencoder and saved configuration.
6. Model evaluation report.
7. Simulated Zero-Trust risk engine.
8. FastAPI backend and API documentation.
9. React security dashboard.
10. Unit, integration and acceptance tests.
11. Setup instructions and dependency files.
12. Final report, presentation and demonstration procedure.

---

# 19. Definition of Done

ResiNet-IDAM is ready for final evaluation when:

- All must-have requirements are implemented.
- Dataset assumptions and preprocessing decisions are documented.
- Model training and evaluation are reproducible.
- The final test set has not been used to tune the threshold.
- Anomaly detection and contextual risk scoring are separate, testable components.
- The API and dashboard operate together locally.
- Events can be traced from model output to dashboard presentation.
- Test results, limitations and simulated assumptions are disclosed.
- A new team member can follow the README and reproduce the demonstration.

---

# 20. Immediate Team Action Plan

Before anyone begins additional development, the team should:

1. Review and approve this blueprint.
2. Assign module owners.
3. Verify the existing HAI preprocessing outputs.
4. Confirm normal-only training eligibility.
5. Freeze the initial feature and event schemas.
6. Implement and evaluate the Autoencoder baseline.
7. Approve the risk formula and simulated context schema.
8. Implement the API and frontend against the shared contracts.
9. Integrate continuously and run acceptance tests.
10. Freeze scope before preparing the final demonstration.

**Governing principle:** Build one reproducible end-to-end prototype first. Add complexity only when the baseline is validated and the additional feature has a measurable purpose.

This blueprint is the team's shared baseline. Changes to scope, architecture, data contracts or acceptance criteria should be documented and agreed upon rather than introduced independently by individual teammates.
