"""
dashboard/app.py

Routing only. No business logic and no page content lives here -
each page module exposes show(), and this file just decides which
one to call based on the sidebar selection.
"""

import os
import sys

import streamlit as st

# project root on sys.path so both `src` and `dashboard` are importable
# as packages, regardless of the working directory this is launched from
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from dashboard.theme import inject_theme
from dashboard.pages import (
    dashboard as dashboard_page,
    prediction,
    analytics,
    explainability,
    model_performance,
    settings,
    about,
)

st.set_page_config(
    page_title="PredictWise",
    page_icon="🏭",
    layout="wide",
)

inject_theme()

PAGES = {
    "Dashboard": dashboard_page,
    "AI Prediction": prediction,
    "Analytics": analytics,
    "Explainability": explainability,
    "Model Performance": model_performance,
    "Settings": settings,
    "About": about,
}

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🏭 PredictWise</div>', unsafe_allow_html=True)
    selected = st.radio(
        "Navigate",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )

PAGES[selected].show()