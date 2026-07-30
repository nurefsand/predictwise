"""
dashboard/pages/settings.py

Session preferences, real model/dataset info, and system status.
No mock charts, no invented metrics - every value here is either a
live UI preference (stored in st.session_state) or something read
directly from the real model file, the real dataset, or the real
src/business_rules.py functions.

Card sections use dashboard.layout.section() - the shared layout
system - so this page no longer needs its own page-local CSS for
card styling.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.theme import (
    HEALTHY, WARNING, CRITICAL, BLUE, TEXT_SECONDARY, TEXT_PRIMARY, BORDER,
    page_title, kpi_card,
)
from dashboard.layout import section, spacer
from dashboard.data import load_scored_fleet
from src.business_rules import calculate_status

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# NOTE: the trained model file is random_forest.pkl, not
# random_forest_model.pkl - using the wrong name here is exactly the
# bug that broke earlier pages, so it's centralized as one constant.
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"

PAGE_NAMES = [
    "Dashboard", "AI Prediction", "Analytics",
    "Explainability", "Model Performance", "Settings", "About",
]

DEFAULT_PREFS = {
    "pref_default_page": "Dashboard",
    "pref_table_rows": 25,
    "pref_auto_refresh": False,
    "pref_refresh_interval": 30,
    "pref_export_recommendations": True,
    "pref_export_confidence": True,
    "pref_export_health_score": True,
    "pref_export_risk_score": True,
    "pref_export_format": "CSV",
}


def _init_prefs():
    for key, default in DEFAULT_PREFS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def show():
    _init_prefs()

    page_title("⚙️", "Settings", "Preferences, model info, and system status for this session")

    _render_info_banner()
    spacer()

    _render_dashboard_preferences()
    spacer()

    _render_model_information()
    spacer()

    _render_risk_thresholds()
    spacer()

    _render_export_settings()
    spacer()

    _render_system_information()


def _render_info_banner():
    st.markdown(
        '<div class="summary-panel">'
        '<div class="summary-line">ℹ️ Most values below are loaded automatically from the '
        "trained model and the live dataset — not editable text, just a live readout of what's "
        "already running.</div></div>",
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------
# DASHBOARD PREFERENCES
# ---------------------------------------------------------------

def _render_dashboard_preferences():
    with section("Dashboard Preferences"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.selectbox(
                "Default landing page", PAGE_NAMES,
                key="pref_default_page",
            )
        with c2:
            st.number_input(
                "Rows in tables", min_value=5, max_value=1000, step=5,
                key="pref_table_rows",
            )
        with c3:
            st.toggle("Auto refresh", key="pref_auto_refresh")
        with c4:
            st.number_input(
                "Refresh interval (s)", min_value=5, max_value=600, step=5,
                key="pref_refresh_interval",
                disabled=not st.session_state["pref_auto_refresh"],
            )


# ---------------------------------------------------------------
# 1. MODEL INFORMATION
# ---------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_model_for_info():
    import joblib
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        import pickle
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)


def _render_model_information():
    with section("Model Information"):
        try:
            model = _load_model_for_info()
            model_ok = True
        except Exception as e:
            model = None
            model_ok = False
            st.warning(f"Couldn't load the model from `{MODEL_PATH}`: {e}")

        df = load_scored_fleet()

        c1, c2, c3, c4 = st.columns(4)
        kpi_card(c1, "🌲", "Algorithm", "Random Forest", BLUE, "rgba(59,130,246,0.14)")
        kpi_card(c2, "🏭", "Dataset Size", f"{len(df):,}", HEALTHY, "rgba(34,197,94,0.14)", "machines")

        if model_ok and MODEL_PATH.exists():
            mtime = pd.Timestamp(MODEL_PATH.stat().st_mtime, unit="s")
            kpi_card(c3, "🕒", "Last Trained", mtime.strftime("%d %b %Y"), WARNING,
                      "rgba(245,158,11,0.14)", "file timestamp*")
        else:
            kpi_card(c3, "🕒", "Last Trained", "Unavailable", TEXT_SECONDARY, "rgba(148,163,184,0.14)")

        n_estimators = getattr(model, "n_estimators", None) if model_ok else None
        kpi_card(c4, "🔢", "Trees", str(n_estimators) if n_estimators else "N/A",
                  BLUE, "rgba(59,130,246,0.14)", "n_estimators")

        st.markdown(
            "<div class='section-subtitle' style='margin-top:8px; margin-bottom:0;'>"
            "*No formal model-version log exists yet — this is the model file's last-modified "
            "timestamp, not a recorded training date.</div>",
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------
# 2. RISK THRESHOLDS
# ---------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _infer_risk_thresholds() -> pd.DataFrame:
    """Does not hardcode any threshold. Calls the real
    calculate_status() from src/business_rules.py across the full
    0-100 range and groups the results, so whatever the function
    actually does today is what gets shown."""
    scores = list(range(0, 101))
    statuses = [calculate_status(s) for s in scores]
    frame = pd.DataFrame({"score": scores, "status": statuses})
    ranges = frame.groupby("status")["score"].agg(["min", "max"]).reset_index()
    ranges.columns = ["Status", "Min", "Max"]
    order = {"Healthy": 0, "Warning": 1, "Critical": 2}
    ranges["_order"] = ranges["Status"].map(order).fillna(99)
    return ranges.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _render_risk_thresholds():
    with section("Risk Thresholds", "Detected live from src/business_rules.py — not hardcoded"):
        try:
            thresholds = _infer_risk_thresholds()
            status_colors = {"Healthy": HEALTHY, "Warning": WARNING, "Critical": CRITICAL}
            status_icons = {"Healthy": "🟢", "Warning": "🟡", "Critical": "🔴"}

            cols = st.columns(len(thresholds))
            for col, (_, row) in zip(cols, thresholds.iterrows()):
                color = status_colors.get(row["Status"], BLUE)
                icon = status_icons.get(row["Status"], "🎯")
                kpi_card(col, icon, row["Status"], f'{int(row["Min"])}–{int(row["Max"])}',
                          color, "rgba(148,163,184,0.14)", "Risk Score range")
        except Exception as e:
            st.warning(
                f"Couldn't read thresholds from src/business_rules.py: {e}. "
                "Check that calculate_status(score) accepts a plain number 0-100."
            )


# ---------------------------------------------------------------
# 3. EXPORT SETTINGS
# ---------------------------------------------------------------

def _render_export_settings():
    with section("Export Settings"):
        left, right = st.columns([2, 1])

        with left:
            st.checkbox("Include recommendations", key="pref_export_recommendations")
            st.checkbox("Include confidence score", key="pref_export_confidence")
            st.checkbox("Include health score", key="pref_export_health_score")
            st.checkbox("Include risk score", key="pref_export_risk_score")

        with right:
            st.radio("Format", ["CSV", "Excel"], key="pref_export_format")


# ---------------------------------------------------------------
# 4. SYSTEM INFORMATION
# ---------------------------------------------------------------

def _render_system_information():
    with section("System Information"):
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        st_version = st.__version__

        try:
            _load_model_for_info()
            model_status, model_bg = "Loaded", "rgba(34,197,94,0.14)"
            model_color = HEALTHY
        except Exception:
            model_status, model_bg, model_color = "Failed", "rgba(239,68,68,0.14)", CRITICAL

        try:
            load_scored_fleet()
            data_status, data_bg = "Loaded", "rgba(34,197,94,0.14)"
            data_color = HEALTHY
        except Exception:
            data_status, data_bg, data_color = "Failed", "rgba(239,68,68,0.14)", CRITICAL

        c1, c2, c3, c4 = st.columns(4)
        kpi_card(c1, "🐍", "Python", py_version, BLUE, "rgba(59,130,246,0.14)")
        kpi_card(c2, "🎈", "Streamlit", st_version, BLUE, "rgba(59,130,246,0.14)")
        kpi_card(c3, "🌲", "Model", model_status, model_color, model_bg)
        kpi_card(c4, "🗂️", "Dataset", data_status, data_color, data_bg)