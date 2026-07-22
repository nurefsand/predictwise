"""
dashboard/pages/settings.py

Session preferences, real model/dataset info, and system status.
No mock charts, no invented metrics - every value here is either a
live UI preference (stored in st.session_state) or something read
directly from the real model file, the real dataset, or the real
src/business_rules.py functions.

Layout note: the brief for this revision asked for four sections
(Model Information, Risk Thresholds, Export Settings, System
Information). Dashboard Preferences (default page, table rows, auto
refresh) isn't in that list, but "keep all current functionality"
means it can't just disappear - it's kept as a compact section right
after the top banner, ahead of the four requested ones.
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

# Page-scoped spacing/card tweaks only - does not touch dashboard/theme.py,
# so no other page is affected.
#
# IMPORTANT: sections that hold plain native widgets (checkboxes, radio,
# selectbox, number_input, toggle) must use st.container(border=True,
# key=...) below, NOT the "open a <div class='chart-card'> via one
# st.markdown call, then close it with a second st.markdown call"
# pattern used elsewhere in this app. That split-div pattern only
# *looks* like it wraps everything because charts/kpi_card divs paint
# their own matching background color - it doesn't actually nest
# separate Streamlit elements inside one real container. For plain
# widgets with no background of their own, that gap shows through as
# empty page background, which is exactly the big blank area in the
# screenshot. st.container(border=True, key=...) is the real fix: it
# produces one genuine DOM container that actually encloses everything
# placed inside its `with` block.
_COMPACT_CSS = """
<style>
.settings-gap { height: 10px; }

.settings-section-title {
    display: block;
    margin-bottom: 4px;
}

div[class*="st-key-settings_card_"] {
    background: #171b22;
    border: 1px solid #2b313d !important;
    border-radius: 10px;
    padding: 16px 18px 10px 18px;
}
</style>
"""


def _init_prefs():
    for key, default in DEFAULT_PREFS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def show():
    _init_prefs()
    st.markdown(_COMPACT_CSS, unsafe_allow_html=True)

    page_title("⚙️", "Settings", "Preferences, model info, and system status for this session")

    _render_info_banner()
    st.markdown("<div class='settings-gap'></div>", unsafe_allow_html=True)

    _render_dashboard_preferences()
    st.markdown("<div class='settings-gap'></div>", unsafe_allow_html=True)

    _render_model_information()
    st.markdown("<div class='settings-gap'></div>", unsafe_allow_html=True)

    _render_risk_thresholds()
    st.markdown("<div class='settings-gap'></div>", unsafe_allow_html=True)

    _render_export_settings()
    st.markdown("<div class='settings-gap'></div>", unsafe_allow_html=True)

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
# DASHBOARD PREFERENCES (kept, tightened)
# ---------------------------------------------------------------

def _render_dashboard_preferences():
    with st.container(border=True, key="settings_card_prefs"):
        st.markdown(
            '<div class="chart-title settings-section-title">Dashboard Preferences</div>'
            '<div style="height:6px"></div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.session_state["pref_default_page"] = st.selectbox(
                "Default landing page", PAGE_NAMES,
                index=PAGE_NAMES.index(st.session_state["pref_default_page"]),
            )
        with c2:
            st.session_state["pref_table_rows"] = st.number_input(
                "Rows in tables", min_value=5, max_value=1000,
                value=st.session_state["pref_table_rows"], step=5,
            )
        with c3:
            st.session_state["pref_auto_refresh"] = st.toggle(
                "Auto refresh", value=st.session_state["pref_auto_refresh"],
            )
        with c4:
            st.session_state["pref_refresh_interval"] = st.number_input(
                "Refresh interval (s)", min_value=5, max_value=600,
                value=st.session_state["pref_refresh_interval"], step=5,
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
    with st.container(border=True, key="settings_card_model"):
        st.markdown('<div class="chart-title settings-section-title">Model Information</div>', unsafe_allow_html=True)

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
            "<div class='chart-note' style='margin-top:6px;'>"
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
    with st.container(border=True, key="settings_card_thresholds"):
        st.markdown(
            '<div class="chart-title settings-section-title">Risk Thresholds</div>'
            '<div class="chart-note">Detected live from src/business_rules.py — not hardcoded</div>'
            '<div style="height:10px"></div>',
            unsafe_allow_html=True
        )

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
    with st.container(border=True, key="settings_card_export"):
        st.markdown(
            '<div class="chart-title settings-section-title">Export Settings</div>'
            '<div style="height:6px"></div>',
            unsafe_allow_html=True
        )

        left, right = st.columns([2, 1])

        with left:
            st.session_state["pref_export_recommendations"] = st.checkbox(
                "Include recommendations", value=st.session_state["pref_export_recommendations"])
            st.session_state["pref_export_confidence"] = st.checkbox(
                "Include confidence score", value=st.session_state["pref_export_confidence"])
            st.session_state["pref_export_health_score"] = st.checkbox(
                "Include health score", value=st.session_state["pref_export_health_score"])
            st.session_state["pref_export_risk_score"] = st.checkbox(
                "Include risk score", value=st.session_state["pref_export_risk_score"])

        with right:
            st.session_state["pref_export_format"] = st.radio(
                "Format",
                ["CSV", "Excel"],
                index=["CSV", "Excel"].index(st.session_state["pref_export_format"]),
            )


# ---------------------------------------------------------------
# 4. SYSTEM INFORMATION
# ---------------------------------------------------------------

def _render_system_information():
    with st.container(border=True, key="settings_card_system"):
        st.markdown(
            '<div class="chart-title settings-section-title">System Information</div>'
            '<div style="height:6px"></div>',
            unsafe_allow_html=True
        )

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