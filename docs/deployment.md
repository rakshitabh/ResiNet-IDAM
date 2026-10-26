# Deployment

## Local Windows run

```cmd
py -3.13 -m venv .venv-ae
.venv-ae\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application runs at `http://localhost:8501`.

## Required artifacts

The OT detector requires:

```text
models/autoencoder.keras
models/anomaly_threshold.json
data/processed/hai-test-scaled.csv
reports/baseline_metrics.json
reports/dataset-profile.json
```

The IT detector becomes available when its deployment bundle is supplied:

```text
data/processed/it_realistic/metadata.json
models/it_realistic/random_forest_all.joblib
```

Build that bundle from the supplied IT CSV with:

```cmd
python scripts\prepare_ids2025_deployment.py
```

The command reads `data/processed/IDS2025.csv`, removes exact duplicates,
fits median imputation and scaling on the training partition only, trains the
Random Forest classifier, and writes the metadata and preprocessing artifacts
under `data/processed/it_realistic/` and `models/it_realistic/`.

If the IT bundle is absent, the application starts in OT-only mode and reports
the missing capability in Administration instead of failing during startup.

## Optional REST adapter

```cmd
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

The REST adapter uses the same application service and event database as
Streamlit.
