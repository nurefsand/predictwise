"""
worker/score_worker.py

Phase 3 of the "make PredictWise live" roadmap: reads new, not-yet-
scored rows from the simulator's `readings` table, scores them with
the EXISTING src/pipeline.py - the exact same score_dataframe() the
dashboard has always used, completely unchanged - and writes the
result into `scored_readings`.

This is a standalone process. Run it in its own terminal, alongside
the simulator:

    python worker/score_worker.py

After this, the dashboard never scores live data itself; it only
reads already-scored rows via dashboard.data.load_live_fleet().
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.db import get_connection, init_db, init_scored_table, READINGS_TABLE, SCORED_TABLE
from src.pipeline import score_dataframe

POLL_SECONDS = 5

OUTPUT_COLUMNS = [
    "reading_id", "timestamp", "machine_id", "Type",
    "Air temperature", "Process temperature", "Rotational speed",
    "Torque", "Tool wear", "Risk Score", "Confidence",
    "Status", "Health Score", "Recommendation",
]


def _last_scored_reading_id(conn) -> int:
    row = conn.execute(f"SELECT MAX(reading_id) FROM {SCORED_TABLE}").fetchone()
    return row[0] if row and row[0] is not None else 0


def _fetch_new_readings(conn, since_id: int) -> pd.DataFrame:
    return pd.read_sql(
        f"SELECT * FROM {READINGS_TABLE} WHERE id > ? ORDER BY id",
        conn, params=(since_id,),
    )


def run():
    init_db()
    init_scored_table()
    conn = get_connection()

    print("🧠 PredictWise scoring worker started.")
    print(f"Polling '{READINGS_TABLE}' every {POLL_SECONDS}s, writing to '{SCORED_TABLE}' — Ctrl+C to stop.\n")

    try:
        while True:
            since_id = _last_scored_reading_id(conn)
            new_rows = _fetch_new_readings(conn, since_id)

            if len(new_rows) > 0:
                # score_dataframe() is completely unchanged from what
                # dashboard/data.py has always called - the worker
                # just calls it on a different (live) input.
                scored = score_dataframe(new_rows)
                scored["reading_id"] = new_rows["id"].values
                scored[OUTPUT_COLUMNS].to_sql(SCORED_TABLE, conn, if_exists="append", index=False)
                print(f"scored {len(new_rows)} new readings (up to reading_id={int(new_rows['id'].max())})")
            else:
                print("no new readings yet...")

            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nWorker stopped.")
    finally:
        conn.close()


if __name__ == "__main__":
    run()