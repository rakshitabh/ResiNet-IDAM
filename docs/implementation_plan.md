# ResiNet-IDAM — Implementation Execution Plan
**Version:** 1.0  
**Date:** 23 October 2026  
**Deployment target:** Local laptop  
**Implementation strategy:** Data-first, modular, test-driven development

## 1. Current Project State

The project has a defined scope, architecture, technology stack and phased roadmap.

The following dataset files have been prepared:

- `hai-train-scaled.csv`
- `hai-test-scaled.csv`
- `standard_scaler.joblib` — expected from the preprocessing pipeline; verify its actual location.

The preparation guide reports 280,800 training records and 54,000 test records, with 66 selected process features.

**Important unresolved assumption:** The training CSV does not contain labels. The team must verify whether it contains only normal observations before using it to train a normal-behavior Autoencoder.

Do not assume that prepared data is automatically validated simply because the CSV files exist.

## 2. Execution Strategy

Implement the project in the following order:

1. Repository and environment verification.
2. Dataset audit and feature-schema freeze.
3. Baseline Autoencoder training.
4. Threshold selection and model evaluation.
5. Simulated Zero-Trust risk engine.
6. Backend API and event storage.
7. Security dashboard.
8. Integration testing and final demonstration.

Each phase must satisfy its exit criteria before the team proceeds to dependent work.

---

## Phase 0 — Repository and Environment Verification

**Objective:** Establish a reproducible development environment for the entire team.

### Tasks

1. Review the existing repository and identify completed scripts.
2. Confirm that the two prepared CSV files are available.
3. Verify Python, Git and the current virtual environment.
4. Install or pin the necessary dependencies.
5. Create the agreed project directories.
6. Record the setup procedure in the README.

### Initial directory structure

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
│   └── api/
├── frontend/
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

Reuse existing files and directories where possible.

### Deliverables

- Verified Python environment.
- Confirmed dataset paths.
- Updated dependency file.
- Documented setup commands.

### Exit criteria

A teammate can clone the repository, follow the setup instructions and locate the required dataset files.

---

## Phase 1 — Dataset Audit and Validation

**Objective:** Understand the actual dataset before model training.

### Task 1.1 — Inspect the CSV files

Create:

`src/data/inspect_dataset.py`

The script must report:

- Number of rows and columns.
- Column names and data types.
- Missing values.
- Duplicate timestamps and rows.
- Constant-valued features.
- Minimum, maximum, mean and standard deviation of numerical features.
- Test label distribution.
- Timestamp range and sampling intervals.

### Task 1.2 — Validate feature consistency

Create:

`src/data/validate_dataset.py`

Verify that:

- The training and test files contain the expected columns.
- Both use the same 66 process features in the same order.
- Timestamp and label columns are excluded from model input.
- Numerical values are finite.
- Test labels contain only the expected classes.
- The scaler was fitted on training data only.
- Features that vary in testing but are constant in training are investigated.

### Task 1.3 — Verify training data semantics

Consult the original HAI dataset documentation and the preprocessing scripts.

Determine whether the prepared training file contains exclusively normal observations. If the training file includes attacks, identify a trustworthy way to construct the normal training subset.

Do not infer that the data is normal-only merely because the file is named `hai-train`.

### Task 1.4 — Document findings

Create:

`docs/dataset-report.md`

Include the dataset source, file dimensions, feature list, label distribution, preprocessing steps, unresolved assumptions and validation results.

### Deliverables

- Dataset inspection report.
- Validated feature schema.
- Confirmed training-data assumptions.
- Documented preprocessing procedure.

### Exit criteria

The team can explain what the model receives as input, what information is reserved for evaluation, and why the selected training samples are suitable.

---

## Phase 2 — Baseline Anomaly Detector

**Objective:** Train a reproducible Autoencoder that produces an anomaly score for every valid observation.

### Task 2.1 — Define the model

Create:

`src/detection/train_autoencoder.py`

Use TensorFlow/Keras for the initial dense Autoencoder.

The initial model may use an encoder with 32, 16 and 8 units, followed by a decoder with 16, 32 and 66 output units. Treat this architecture as a baseline to evaluate, not an established optimum.

### Task 2.2 — Define training and validation

1. Use verified normal training observations.
2. Split training data into training and validation portions.
3. Respect temporal ordering when appropriate.
4. Train the model to reconstruct its input features.
5. Use early stopping to limit unnecessary training.
6. Record the model configuration and training history.

### Task 2.3 — Calculate reconstruction errors

For each observation, calculate mean squared reconstruction error:

`MSE = mean((original_features - reconstructed_features)²)`

Higher reconstruction error indicates a greater deviation from the patterns learned by the model.

### Task 2.4 — Select the threshold

Create:

`src/detection/select_threshold.py`

Choose a threshold using verified normal validation observations. Start with a documented percentile-based threshold and evaluate its false-positive behavior.

Do not use final test labels to optimize the threshold.

### Task 2.5 — Save model artifacts

Store:

- `models/autoencoder.keras`
- `models/feature_columns.json`
- `models/anomaly_threshold.json`
- `models/training_config.json`

### Deliverables

- Trained Autoencoder.
- Saved threshold.
- Feature configuration.
- Training history and reproducibility information.

### Exit criteria

The model can be loaded independently of training and can calculate reconstruction errors for previously unseen, schema-compatible observations.

---

## Phase 3 — Evaluation and Error Analysis

**Objective:** Determine whether the baseline detector identifies attack-associated observations effectively.

Create:

`src/detection/evaluate_model.py`

### Evaluation procedure

1. Load the saved model and threshold.
2. Read the test features without passing the label column to the model.
3. Calculate reconstruction errors.
4. Generate anomaly predictions.
5. Compare predictions with ground-truth labels in a separate evaluation step.
6. Generate and save the evaluation report.

### Required metrics

- Confusion matrix.
- Precision.
- Recall.
- F1-score.
- False-positive rate.
- PR-AUC.
- ROC-AUC.

### Required outputs

- `reports/baseline_metrics.json`
- `reports/test_anomaly_scores.csv`
- `docs/model-evaluation.md`

The report must describe the train/validation/test split, threshold-selection method, model configuration and limitations.

### Exit criteria

The team can reproduce the baseline results and explain false positives, missed attacks and the effects of class imbalance.

Do not claim that the system guarantees detection of real-world zero-day attacks. Report what the benchmark experiment actually demonstrates.

---

## Phase 4 — Simulated Zero-Trust Risk Engine

**Objective:** Convert anomaly results into contextual security risk assessments.

Create:

`src/risk/risk_scoring.py`

### Context fields

- `identity_id`
- `device_id`
- `asset_id`
- `identity_trust`
- `device_trust`
- `asset_criticality`
- `anomaly_severity`
- `behavior_deviation`

Use simulated identities, devices and asset attributes. Document how the simulated values are generated.

### Initial scoring formula

`risk = 100 × (0.40 × anomaly_severity + 0.20 × behavior_deviation + 0.20 × (1 − identity_trust) + 0.10 × (1 − device_trust) + 0.10 × asset_criticality)`

All inputs must be normalized between 0 and 1.

### Risk bands

| Score | Risk level |
|---|---|
| 0–29 | Low |
| 30–59 | Medium |
| 60–79 | High |
| 80–100 | Critical |

### Testing

Verify:

- Valid scores remain between 0 and 100.
- Higher anomaly severity does not lower risk when other inputs are fixed.
- Lower identity or device trust increases risk when other inputs are fixed.
- Invalid values are rejected.
- Identical inputs produce identical results.

### Deliverables

- Risk-scoring module.
- Simulated context configuration.
- Risk policy documentation.
- Unit tests.

### Exit criteria

Each event receives a deterministic risk score, risk classification and component-level explanation.

---

## Phase 5 — Backend API and Event Storage

**Objective:** Make the model and risk engine accessible to the dashboard.

### Technology

- FastAPI.
- Uvicorn.
- SQLite.
- Pydantic request and response schemas.

### Required endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Application health |
| `GET /api/summary` | Event and risk summaries |
| `GET /api/events` | List and filter events |
| `GET /api/events/{event_id}` | Inspect an individual event |
| `GET /api/model/info` | Model version and threshold |
| `GET /api/model/metrics` | Evaluation metrics |
| `POST /api/events/score` | Score a valid observation |

### Event record

Each stored event should include:

- Event ID.
- Timestamp.
- Asset ID.
- Anomaly score.
- Threshold and anomaly decision.
- Simulated identity/device context.
- Risk score and risk level.
- Risk component breakdown.
- Event review status.

Ground-truth labels must not be required for inference requests.

### Deliverables

- Working API.
- SQLite event storage.
- API documentation.
- API and integration tests.

### Exit criteria

A valid input produces an event that can be retrieved through the API with its anomaly and risk results.

---

## Phase 6 — Security Dashboard

**Objective:** Present the results in an understandable interface.

### Technology

- React.
- Vite.
- Recharts.

### Required pages

**Overview**
- Total events.
- Anomalies detected.
- Risk distribution.
- Recent high-risk events.

**Event explorer**
- Timestamp.
- Asset.
- Anomaly score.
- Risk score.
- Severity.
- Review status.

**Event details**
- Reconstruction error and threshold.
- Simulated identity and device trust.
- Asset criticality.
- Risk score breakdown.
- Event metadata.

**Model evaluation**
- Precision, recall and F1-score.
- False-positive rate.
- Confusion matrix.
- PR-AUC and ROC-AUC.

### Deliverables

- Functional React dashboard.
- API integration.
- Filters and event detail views.
- Loading, empty and error states.

### Exit criteria

A reviewer can inspect an event from its timestamp through its anomaly result and contextual risk classification.

---

## Phase 7 — Integration and Acceptance Testing

**Objective:** Verify that the complete prototype works as one system.

### End-to-end scenario

1. Load a prepared HAI observation.
2. Validate the feature schema.
3. Generate the anomaly score.
4. Apply the saved threshold.
5. Attach simulated identity/device context.
6. Calculate the risk score.
7. Store the event.
8. Retrieve the event through the API.
9. Display it in the dashboard.
10. Review model evaluation metrics separately.

### Final acceptance criteria

- All must-have requirements are implemented.
- Dataset and model assumptions are documented.
- Model inference works without retraining.
- Ground-truth labels are excluded from inference inputs.
- Risk scoring passes unit tests.
- API and dashboard integration works.
- Evaluation results are reproducible.
- The local demonstration runs from documented instructions.

### Deliverables

- Acceptance test report.
- Final technical documentation.
- Demonstration script.
- Project presentation.
- Reproducible local setup instructions.

---

## Phase 8 — Optional Research Extensions

Only after the core prototype passes acceptance testing:

1. Compare the Autoencoder against Isolation Forest.
2. Investigate BiLSTM-based temporal detection.
3. Implement controlled retraining with validation and rollback.
4. Add simulated IT events and richer contextual relationships.
5. Add MQTT or Kafka only if a streaming demonstration is required.

Each extension must have a defined objective, evaluation method and measurable benefit.

---

# 3. Team Execution Rules

1. Assign one accountable owner to each workstream.
2. Keep dataset preparation, model training, risk scoring, API and frontend modules separate.
3. Agree on data schemas before parallel development.
4. Use feature branches and reviewed pull requests.
5. Add tests with each module.
6. Record changes to model configuration, thresholds and risk weights.
7. Integrate incrementally.
8. Do not change shared interfaces without informing the other module owners.
9. Keep large datasets and generated model artifacts out of Git unless the repository explicitly requires them.
10. Maintain a single task tracker with owner, status, dependency and acceptance criteria.

# 4. Immediate Next Actions

The team should begin with these tasks in order:

1. Verify the actual repository structure and existing scripts.
2. Run the dataset inspection and validation scripts.
3. Confirm whether training data contains only normal observations.
4. Freeze the ordered list of 66 model features.
5. Document the dataset findings.
6. Implement the baseline Autoencoder.
7. Evaluate and document the baseline.
8. Proceed to risk scoring only after the model outputs are reliable.

**First milestone:** A validated dataset and a reproducible baseline anomaly detector, with saved model artifacts and an evaluation report.

The dashboard, streaming infrastructure and advanced models are downstream work—not prerequisites for the first milestone.
