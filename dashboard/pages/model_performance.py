"""
dashboard/pages/model_performance.py

Model Performance page - placeholder. Will show accuracy, precision,
recall, F1, ROC curve, confusion matrix, classification report, and
cross-validation results, using the held-out evaluation data/logic
already produced during training (src/train_model.py).
"""

import streamlit as st
from dashboard.theme import page_title


def show():
    page_title("📈", "Model Performance", "How well the Random Forest model performs")
    st.info("Model Performance is queued after Explainability.")