"""
dashboard/pages/explainability.py

Explainability page - placeholder. Will reuse src/explainability.py
for SHAP values (global + local), feature importance, and per-machine
"what pushed this risk up/down" breakdowns.
"""

import streamlit as st
from dashboard.theme import page_title


def show():
    page_title("🧬", "Explainability", "Why the model predicts what it predicts")
    st.info("Explainability is queued after Analytics.")