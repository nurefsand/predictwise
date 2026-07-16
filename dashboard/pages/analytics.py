"""
dashboard/pages/analytics.py

Fleet analytics page - placeholder. Will be built next: failure by
machine type, health/risk distributions, sensor charts, correlation
heatmap, feature importance. Will reuse dashboard.data.load_scored_fleet()
for data and src/explainability.py for feature importance.
"""

import streamlit as st
from dashboard.theme import page_title


def show():
    page_title("📊", "Analytics", "Fleet-wide trends and correlations")
    st.info("Analytics is next in the build queue — coming right after AI Prediction is confirmed working.")