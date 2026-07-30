"""
dashboard/data.py

Streamlit-specific caching wrapper around src/pipeline.py.

Why this lives in dashboard/ and not src/: src/pipeline.py stays pure
pandas/sklearn code with no Streamlit dependency, so it can be tested
or reused outside the app. This file is the only place that knows
about st.cache_data - it's a UI-layer concern (avoiding recompute on
every rerun), not business logic.
"""

import streamlit as st
import pandas as pd

from src.pipeline import score_dataframe
from simulator.db import get_connection, SCORED_TABLE

DATA_PATH = "data/predictive_maintenance.csv"


@st.cache_data(show_spinner="Loading fleet data...")
def load_scored_fleet(path: str = DATA_PATH) -> pd.DataFrame:
    """Loads the base fleet CSV and returns it fully scored
    (Risk Score, Confidence, Status, Health Score, Recommendation).

    This is the historical reference dataset - Analytics,
    Explainability, Model Performance, and About all use this, since
    they need a large, stable dataset (and, for Model Performance,
    real ground-truth labels) rather than a live feed that starts
    empty and only grows slowly."""
    raw = pd.read_csv(path)
    return score_dataframe(raw)


@st.cache_data(ttl=10, show_spinner=False)
def load_live_fleet() -> pd.DataFrame:
    """Reads the latest scored reading per machine from the live
    pipeline (simulator/generate_readings.py -> worker/score_worker.py
    -> scored_readings table). This is what the Dashboard page shows
    when the live pipeline is running.

    Returns an empty DataFrame - never raises - if the live database
    or table doesn't exist yet (e.g. the simulator/worker haven't been
    started). Callers should check `.empty` and fall back to
    load_scored_fleet() in that case.

    Cached for only 10 seconds (not indefinitely like
    load_scored_fleet) so the dashboard actually reflects new data as
    the worker produces it."""
    try:
        conn = get_connection()
        df = pd.read_sql(f"""
            SELECT sr.* FROM {SCORED_TABLE} sr
            INNER JOIN (
                SELECT machine_id, MAX(reading_id) AS max_id
                FROM {SCORED_TABLE}
                GROUP BY machine_id
            ) latest
            ON sr.machine_id = latest.machine_id AND sr.reading_id = latest.max_id
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()