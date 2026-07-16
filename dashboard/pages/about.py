"""
dashboard/pages/about.py

About page - placeholder. Will describe the project, dataset
(AI4I 2020), model (Random Forest), libraries used, authors, license.
Pure static content, no src/ dependency.
"""

import streamlit as st
from dashboard.theme import page_title


def show():
    page_title("ℹ️", "About", "Project, dataset, and model information")
    st.info("About is queued last.")