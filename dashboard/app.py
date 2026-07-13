import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os


# src klasörünü görebilmesi için
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.feature_engineering import create_features
from src.preprocessing import prepare_data
from src.prediction import predict_probability
from src.business_rules import (
    calculate_status,
    calculate_health_score,
    maintenance_recommendation
)

st.set_page_config(
    page_title="PredictWise",
    page_icon="🏭",
    layout="wide"
)

# -------------------------
# DATA
# -------------------------

df = pd.read_csv("data/predictive_maintenance.csv")

df = create_features(df)

X = prepare_data(df)

risk_scores = predict_probability(X)

df["Risk Score"] = risk_scores * 100

df["Status"] = df["Risk Score"].apply(calculate_status)

df["Health Score"] = df["Risk Score"].apply(calculate_health_score)

df["Recommendation"] = df["Risk Score"].apply(
    maintenance_recommendation
)

# -------------------------
# HEADER
# -------------------------

st.title("🏭 PredictWise")

st.caption(
    "AI-Powered Predictive Maintenance Decision Support System"
)

st.divider()

# -------------------------
# KPI
# -------------------------

healthy = (df["Status"] == "Healthy").sum()

warning = (df["Status"] == "Warning").sum()

critical = (df["Status"] == "Critical").sum()

avg_health = df["Health Score"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "🟢 Healthy",
    healthy
)

col2.metric(
    "🟡 Warning",
    warning
)

col3.metric(
    "🔴 Critical",
    critical
)

col4.metric(
    "💙 Avg Health",
    f"{avg_health:.1f}"
)

st.divider()

# -------------------------
# CHARTS
# -------------------------

left, right = st.columns(2)

with left:

    status_counts = (
        df["Status"]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = [
        "Status",
        "Count"
    ]

    fig = px.pie(
        status_counts,
        names="Status",
        values="Count",
        title="Machine Status Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    fig = px.histogram(
        df,
        x="Risk Score",
        nbins=30,
        title="Risk Score Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# -------------------------
# FILTER
# -------------------------

selected_status = st.selectbox(
    "Filter Machines",
    [
        "All",
        "Healthy",
        "Warning",
        "Critical"
    ]
)

if selected_status != "All":
    filtered_df = df[df["Status"] == selected_status]
else:
    filtered_df = df

# -------------------------
# TABLE
# -------------------------

st.subheader("Machine Overview")

st.dataframe(
    filtered_df[
        [
            "Type",
            "Air temperature",
            "Process temperature",
            "Rotational speed",
            "Torque",
            "Tool wear",
            "Risk Score",
            "Health Score",
            "Status",
            "Recommendation"
        ]
    ],
    use_container_width=True,
    hide_index=True
)