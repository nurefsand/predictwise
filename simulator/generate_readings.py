"""
simulator/generate_readings.py

Phase 1 of the "make PredictWise live" roadmap: a standalone process
that simulates a live sensor feed for a fixed fleet of virtual
machines, based on realistic baseline values sampled from the real
AI4I 2020 dataset (data/predictive_maintenance.csv).

This is NOT part of the Streamlit app - it never imports from
dashboard/, and the dashboard never imports from here. It only writes
to the shared SQLite database defined in simulator/db.py. Phase 3's
scoring worker will read from that same table and run the EXISTING
src/pipeline.py on it, completely unchanged.

Run it in its own terminal, separate from the dashboard:

    python simulator/generate_readings.py

Stop it any time with Ctrl+C.

Design note on realism: instead of just adding random jitter every
tick (which would make Risk Score bounce around meaninglessly), each
virtual machine accumulates Tool Wear over time - the same way a real
machine's wear only goes up between maintenance events - and
occasionally "resets" once it crosses a threshold, simulating a
maintenance visit. That's what will make the trend chart in Phase 5
actually show something meaningful (rising risk, then a drop after
maintenance) instead of flat noise.
"""

import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.db import get_connection, init_db, READINGS_TABLE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = PROJECT_ROOT / "data" / "predictive_maintenance.csv"

# ---- tunable constants - change these freely, no other code depends on them ----
N_VIRTUAL_MACHINES = 30
TICK_SECONDS = 5                 # how often every machine gets a new reading
NOISE_PCT = 0.015                 # +/- 1.5% gaussian jitter on temp/speed/torque
WEAR_INCREMENT_RANGE = (1.0, 3.0)  # minutes of tool wear added per tick
WEAR_RESET_THRESHOLD = 240.0       # simulated maintenance reset above this


class VirtualMachine:
    """One simulated machine: a stable baseline sensor profile (sampled
    once from a real dataset row) plus cumulative tool wear that grows
    every tick and occasionally resets."""

    def __init__(self, machine_id: str, baseline_row: pd.Series):
        self.machine_id = machine_id
        self.type = baseline_row["Type"]
        self.baseline_air_temp = float(baseline_row["Air temperature"])
        self.baseline_process_temp = float(baseline_row["Process temperature"])
        self.baseline_speed = float(baseline_row["Rotational speed"])
        self.baseline_torque = float(baseline_row["Torque"])
        self.tool_wear = float(baseline_row["Tool wear"])

    def _jitter(self, value: float) -> float:
        return value * (1 + random.gauss(0, NOISE_PCT))

    def tick(self) -> dict:
        self.tool_wear += random.uniform(*WEAR_INCREMENT_RANGE)
        if self.tool_wear > WEAR_RESET_THRESHOLD:
            self.tool_wear = 0.0  # simulated maintenance reset

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "machine_id": self.machine_id,
            "Type": self.type,
            "Air temperature": round(self._jitter(self.baseline_air_temp), 2),
            "Process temperature": round(self._jitter(self.baseline_process_temp), 2),
            "Rotational speed": round(self._jitter(self.baseline_speed), 1),
            "Torque": round(self._jitter(self.baseline_torque), 2),
            "Tool wear": round(self.tool_wear, 1),
        }


def build_virtual_fleet(n: int) -> list:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"Can't find {SOURCE_CSV}. The simulator samples realistic "
            "baseline profiles from the real dataset, so it needs to exist "
            "at that path."
        )
    df = pd.read_csv(SOURCE_CSV)
    sample = df.sample(n=min(n, len(df))).reset_index(drop=True)
    return [VirtualMachine(f"VM-{i:03d}", sample.iloc[i]) for i in range(len(sample))]


def run():
    init_db()
    fleet = build_virtual_fleet(N_VIRTUAL_MACHINES)
    conn = get_connection()

    print(f"🏭 PredictWise simulator started — {len(fleet)} virtual machines, "
          f"one new reading per machine every {TICK_SECONDS}s.")
    print(f"Writing to data/live_readings.db ({READINGS_TABLE}) — Ctrl+C to stop.\n")

    try:
        while True:
            readings = [vm.tick() for vm in fleet]
            batch = pd.DataFrame(readings)
            batch.to_sql(READINGS_TABLE, conn, if_exists="append", index=False)

            sample = readings[0]
            print(f"[{sample['timestamp']}] wrote {len(readings)} readings "
                  f"(e.g. {sample['machine_id']}: Tool wear={sample['Tool wear']} min)")

            time.sleep(TICK_SECONDS)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
    finally:
        conn.close()


if __name__ == "__main__":
    run()