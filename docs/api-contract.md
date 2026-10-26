# ResiNet-IDAM API Contract

Start the local API from the repository root:

```powershell
.\.venv-ae\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

The interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Application health |
| GET | `/api/summary` | Event totals and risk distribution |
| GET | `/api/events` | Event listing with optional `risk_level`, `asset_id`, `review_status`, and `limit` filters |
| GET | `/api/events/{event_id}` | Retrieve one persisted event |
| GET | `/api/model/info` | Model version, feature contract, and threshold |
| GET | `/api/model/metrics` | Measured baseline evaluation metrics |
| GET | `/api/dataset/profile` | Dataset dimensions, measurements, feature groups, and feature dictionary |
| POST | `/api/events/score` | Score and persist one observation |

## Scoring request

```json
{
  "timestamp": "2022-08-12T16:00:01Z",
  "features": {
    "P1_FCV01D": 0.0
  },
  "context": {
    "identity_id": "identity-1",
    "device_id": "device-1",
    "asset_id": "asset-1",
    "identity_trust": 0.8,
    "device_trust": 0.8,
    "asset_criticality": 0.6,
    "behavior_deviation": 0.2
  },
  "review_status": "NEW"
}
```

The `features` object must contain exactly all 66 names from `/api/model/info`, in the same order. It must not contain
`timestamp` or a ground-truth `label`. Values must be finite numbers.

The API derives `anomaly_severity` from the reconstruction error relative to the saved threshold, calculates the
simulated risk score, and persists the event in a local SQLite database. Ground-truth labels are never accepted by the
inference endpoint.

## Error behavior

- `404`: requested event does not exist.
- `422`: malformed request or incompatible feature schema.
- `503`: model or metrics artifacts are unavailable.

The risk context is simulated research data and is not real authentication or authorization evidence.

## Dashboard startup

In a second terminal, start the Vite dashboard:

PowerShell:

```powershell
Set-Location frontend
npm install
npm run dev
```

Command Prompt:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dashboard reads from the API at `http://127.0.0.1:8000/api` by default. Set
`VITE_API_BASE` when the API is hosted elsewhere.
