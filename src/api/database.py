"""SQLite persistence for scored security events."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    anomaly_score REAL NOT NULL,
    threshold REAL NOT NULL,
    anomaly_decision INTEGER NOT NULL,
    risk_score REAL NOT NULL,
    risk_level TEXT NOT NULL,
    review_status TEXT NOT NULL,
    identity_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    context_json TEXT NOT NULL,
    risk_components_json TEXT NOT NULL
);
"""


class EventStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def create(self, event: dict[str, Any]) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events (
                    event_id, timestamp, asset_id, anomaly_score, threshold,
                    anomaly_decision, risk_score, risk_level, review_status,
                    identity_id, device_id, context_json, risk_components_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["event_id"],
                    event["timestamp"],
                    event["asset_id"],
                    event["anomaly_score"],
                    event["threshold"],
                    int(event["anomaly_decision"]),
                    event["risk_score"],
                    event["risk_level"],
                    event["review_status"],
                    event["identity_id"],
                    event["device_id"],
                    json.dumps(event["context"]),
                    json.dumps(event["risk_components"]),
                ),
            )
        return event

    def get(self, event_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return self._deserialize(row) if row else None

    def list(
        self,
        risk_level: str | None = None,
        asset_id: str | None = None,
        review_status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[str | int] = []
        if risk_level:
            clauses.append("risk_level = ?")
            values.append(risk_level)
        if asset_id:
            clauses.append("asset_id = ?")
            values.append(asset_id)
        if review_status:
            clauses.append("review_status = ?")
            values.append(review_status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ?",
                values,
            ).fetchall()
        return [self._deserialize(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            anomalies = connection.execute(
                "SELECT COUNT(*) FROM events WHERE anomaly_decision = 1"
            ).fetchone()[0]
            levels = connection.execute(
                "SELECT risk_level, COUNT(*) FROM events GROUP BY risk_level"
            ).fetchall()
        return {
            "total_events": total,
            "anomalies_detected": anomalies,
            "risk_distribution": {row[0]: row[1] for row in levels},
        }

    @staticmethod
    def _deserialize(row: sqlite3.Row) -> dict[str, Any]:
        event = dict(row)
        event["anomaly_decision"] = bool(event["anomaly_decision"])
        event["context"] = json.loads(event.pop("context_json"))
        event["risk_components"] = json.loads(event.pop("risk_components_json"))
        return event
