"""
ResiNet-IDAM: SQLite Investigation History Helper
-------------------------------------------------
Stores reviewer audit notes, tags, and investigation verdicts for inspected predictions.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "investigation_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS investigations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            domain TEXT NOT NULL,
            record_id TEXT NOT NULL,
            true_label TEXT NOT NULL,
            pred_label TEXT NOT NULL,
            verdict TEXT NOT NULL,
            notes TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

def add_investigation(domain, record_id, true_label, pred_label, verdict, notes, confidence):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO investigations (timestamp, domain, record_id, true_label, pred_label, verdict, notes, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        domain,
        str(record_id),
        str(true_label),
        str(pred_label),
        verdict,
        notes,
        float(confidence) if confidence is not None else 0.0
    ))
    conn.commit()
    conn.close()

def get_investigations():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, domain, record_id, true_label, pred_label, verdict, notes, confidence FROM investigations ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows
