# ResiNet-IDAM Deployment Architecture

ResiNet-IDAM is deployed as one Streamlit application. Streamlit owns the
operator experience; application services own orchestration; detectors,
the risk engine, and the event store are reusable backend components.

```text
Streamlit app.py
    |
    +-- Operations overview
    +-- Detection workspace
    +-- Alert management
    +-- Risk assessment
    +-- Model and data health
    +-- Administration
    |
    +-- DetectionService
          |
          +-- Detector registry (OT HAI / IT IDS2025)
          +-- Zero-Trust risk engine
          +-- SQLite EventStore
```

The FastAPI adapter remains available for machine-to-machine integrations. It
delegates scoring to the same `DetectionService`; it is not a second dashboard.

## Runtime boundaries

- `src/domain/` contains event and detector contracts.
- `src/application/` contains detection orchestration.
- `src/detectors/` contains source-specific inference adapters.
- `src/infrastructure/` contains environment configuration and artifact access.
- `src/api/` contains persistence and the optional REST adapter.
- `app.py` contains only Streamlit navigation and presentation composition.

Generated datasets, trained models, SQLite databases, and local reports are
deployment artifacts. They are supplied through deployment storage and are not
committed to the repository.
