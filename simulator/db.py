"""
simulator/db.py

Shared SQLite connection/schema helper for the live-data pipeline.
Both the simulator (writes new readings) and the future scoring
worker (Phase 3, reads new readings + writes scored results) import
from here, so the database path and table schema live in exactly one
place - not duplicated per script.

Column names match src/pipeline.py's expected raw schema exactly
(Type, Air temperature, Process temperature, Rotational speed,
Torque, Tool wear), so Phase 3 can feed rows from this table straight
into get_feature_matrix()/score_dataframe() with zero renaming.
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "live_readings.db"

READINGS_TABLE = "readings"
SCORED_TABLE = "scored_readings"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """Creates the readings table if it doesn't exist yet. Safe to
    call every time the simulator starts - a no-op if already there."""
    conn = get_connection()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {READINGS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            machine_id TEXT NOT NULL,
            "Type" TEXT NOT NULL,
            "Air temperature" REAL NOT NULL,
            "Process temperature" REAL NOT NULL,
            "Rotational speed" REAL NOT NULL,
            "Torque" REAL NOT NULL,
            "Tool wear" REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def init_scored_table():
    """Creates the scored_readings table if it doesn't exist yet.
    Written by worker/score_worker.py, read by
    dashboard.data.load_live_fleet(). reading_id links each scored row
    back to the raw reading it came from, so the worker can tell which
    rows it hasn't scored yet."""
    conn = get_connection()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCORED_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reading_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            machine_id TEXT NOT NULL,
            "Type" TEXT NOT NULL,
            "Air temperature" REAL NOT NULL,
            "Process temperature" REAL NOT NULL,
            "Rotational speed" REAL NOT NULL,
            "Torque" REAL NOT NULL,
            "Tool wear" REAL NOT NULL,
            "Risk Score" REAL NOT NULL,
            "Confidence" REAL NOT NULL,
            "Status" TEXT NOT NULL,
            "Health Score" REAL NOT NULL,
            "Recommendation" TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()