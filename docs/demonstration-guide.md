# ResiNet-IDAM Demonstration Guide

## Before the demonstration

Use the Python 3.13 environment:

```bat
cd /d D:\resinet-idam\ResiNet-IDAM
.\.venv-ae\Scripts\python.exe -m pip install -r requirements.txt
```

Start the API in one Command Prompt:

```bat
cd /d D:\resinet-idam\ResiNet-IDAM
.\.venv-ae\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

Start the dashboard in a second Command Prompt:

```bat
cd /d D:\resinet-idam\ResiNet-IDAM\frontend
npm run dev
```

Open `http://localhost:5173`.

## Demonstration sequence

1. **Dataset:** Explain that the HAI prepared files contain 66 process features, timestamps, and held-out labels.
2. **Model:** Open the Model evaluation page and show the saved model version, threshold, precision, recall, F1, false-positive rate, ROC-AUC, PR-AUC, and confusion matrix.
3. **Inference boundary:** Explain that the dashboard/API receives only the 66 ordered features. Ground-truth labels are not accepted by the inference endpoint.
4. **Event context:** Use a scored event to show reconstruction error, threshold comparison, and anomaly decision.
5. **Risk assessment:** Open event details and show simulated identity trust, device trust, asset criticality, behavior deviation, and weighted risk components.
6. **Persistence:** Refresh the dashboard and show that the event remains available through SQLite-backed API retrieval.
7. **Limitations:** State that trust context is simulated, benchmark results are not production guarantees, and critical actions are recommendations for human approval only.

## Evidence to capture

- Dashboard overview screenshot.
- Event detail screenshot with risk component breakdown.
- Model evaluation screenshot.
- [Dataset report](./dataset-report.md).
- [Model evaluation report](./model-evaluation.md).
- [Integration test report](./integration-test-report.md).
- [Performance benchmark](../reports/performance-benchmark.json).

## Measured performance

Run the benchmark from the repository root:

```bat
.\.venv-ae\Scripts\python.exe scripts\benchmark_performance.py
```

For a non-destructive inference-only run:

```bat
.\.venv-ae\Scripts\python.exe scripts\benchmark_performance.py --skip-training
```

The benchmark records training duration, inference throughput, per-row inference latency, and process RSS. These are
observations on the development laptop and must not be presented as universal capacity targets.

The measured run produced:

- Training duration: **15.29 seconds**
- Batch inference: **54,000 rows in 0.2826 seconds**
- Throughput: **191,073.54 rows/second**
- Per-row inference latency: **0.0052 ms**
- RSS after model load: **372.16 MB**
- RSS after inference: **406.78 MB**
- Dashboard initial load: **223 ms**

The dashboard load measurement rendered the Overview, Model evaluation, Event explorer, and empty-event states in the
live browser. See [performance-benchmark.json](../reports/performance-benchmark.json) for the complete record.
