"""
dashboard/pages/settings.py

Settings page - placeholder. Will hold theme toggle, prediction
threshold slider, CSV export delimiter, and a reset button. These are
UI/session preferences, so this page will stay pure Streamlit with no
src/ dependency beyond reading the current threshold used elsewhere.
"""

import streamlit as st
from dashboard.theme import page_title


def show():
    page_title("⚙️", "Settings", "Preferences for this session")
    st.info("Settings is queued after Model Performance.")