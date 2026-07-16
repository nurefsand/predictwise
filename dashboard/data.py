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

DATA_PATH = "data/predictive_maintenance.csv"


@st.cache_data(show_spinner="Loading fleet data...")
def load_scored_fleet(path: str = DATA_PATH) -> pd.DataFrame:
    """Loads the base fleet CSV and returns it fully scored
    (Risk Score, Confidence, Status, Health Score, Recommendation)."""
    raw = pd.read_csv(path)
    return score_dataframe(raw)