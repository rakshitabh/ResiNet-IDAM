# Integration and Acceptance Test Report

**Test date:** 23 October 2026  
**Environment:** Windows, Python 3.13.5, TensorFlow 2.21.0, FastAPI, React/Vite

## Automated verification

| Check | Result |
|---|---|
| Python test suite | 12 passed, 4 risk subtests passed |
| Frontend TypeScript compilation | Passed |
| Frontend production build | Passed |
| API health endpoint | HTTP 200 |
| Model metrics endpoint | HTTP 200 |
| API feature-schema rejection | Passed |
| SQLite event persistence tests | Passed |

The FastAPI test suite emits one dependency warning from Starlette about the future `httpx2` package. It does not
affect the current test result.

## End-to-end acceptance scenario

A real first-row observation from `data/processed/hai-test-scaled.csv` was submitted through
`POST /api/events/score` with ground-truth labels excluded from the request.

The flow completed successfully:

1. The 66-feature schema was matched against the saved model contract.
2. The Autoencoder calculated a reconstruction error.
3. The saved threshold produced an anomaly decision.
4. Simulated identity, device, and asset context was attached.
5. The risk engine produced a score and severity.
6. The event was persisted to an isolated SQLite database.
7. The event was retrieved through `GET /api/events/{event_id}`.
8. `GET /api/summary` reported the persisted event.

Observed acceptance result for this sample:

- Anomaly decision: `false`
- Risk level: `MEDIUM`
- Risk score: `41.22`
- Summary total: `1`

This result demonstrates system connectivity, not model quality. The risk level includes simulated context and must not
be interpreted as proof of malicious activity.

## Current acceptance status

All implemented core workflow checks pass. Performance measurements and demonstration evidence are recorded in
[performance-benchmark.json](../reports/performance-benchmark.json) and
[demonstration-guide.md](./demonstration-guide.md).
