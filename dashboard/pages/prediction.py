"""
dashboard/pages/prediction.py

AI Prediction page: score one machine via manual input, or a batch
of machines via CSV upload. All ML logic is delegated to
src/pipeline.py (which only orchestrates the existing
feature_engineering, preprocessing, prediction and business_rules
modules) - nothing here re-implements any of that.
"""

import streamlit as st
import pandas as pd

from dashboard.theme import (
    page_title, badge_html, status_color, kpi_card,
    STATUS_COLORS, TEXT_SECONDARY,
)
from src.pipeline import score_dataframe

RAW_COLUMNS = [
    "Type", "Air temperature", "Process temperature",
    "Rotational speed", "Torque", "Tool wear",
]


def show():
    page_title("🔮", "AI Prediction", "Score a machine and get a maintenance recommendation")

    mode = st.radio(
        "Input mode",
        ["Manual entry", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if mode == "Manual entry":
        _manual_entry()
    else:
        _csv_upload()


def _manual_entry():
    with st.form("manual_prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            machine_type = st.selectbox("Type", ["L", "M", "H"])
            air_temp = st.number_input("Air temperature [K]", value=300.0, step=0.1)
        with c2:
            process_temp = st.number_input("Process temperature [K]", value=310.0, step=0.1)
            rot_speed = st.number_input("Rotational speed [rpm]", value=1500, step=10)
        with c3:
            torque = st.number_input("Torque [Nm]", value=40.0, step=0.5)
            tool_wear = st.number_input("Tool wear [min]", value=0, step=1)

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    if not submitted:
        return

    raw = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature": air_temp,
        "Process temperature": process_temp,
        "Rotational speed": rot_speed,
        "Torque": torque,
        "Tool wear": tool_wear,
    }])

    scored = score_dataframe(raw)
    _render_single_result(scored.iloc[0])


def _render_single_result(row: pd.Series):
    fg, _ = status_color(row["Status"])

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "🎯", "Predicted Status", row["Status"], fg, "rgba(148,163,184,0.14)")
    kpi_card(c2, "⚠️", "Risk Score", f'{row["Risk Score"]:.1f}', fg, "rgba(148,163,184,0.14)")
    kpi_card(c3, "💙", "Health Score", f'{row["Health Score"]:.1f}', fg, "rgba(148,163,184,0.14)")
    kpi_card(c4, "📊", "Confidence", f'{row["Confidence"]:.1f}%', fg, "rgba(148,163,184,0.14)")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="summary-panel"><div class="summary-title">Recommendation</div>'
        f'<div class="summary-line">{badge_html(row["Status"])} &nbsp; {row["Recommendation"]}</div></div>',
        unsafe_allow_html=True,
    )


def _csv_upload():
    uploaded = st.file_uploader(
        "Upload a CSV with columns: " + ", ".join(RAW_COLUMNS),
        type=["csv"],
    )

    if uploaded is None:
        st.caption("Waiting for a file. The CSV must use the same column names as the training data.")
        return

    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    scored = score_dataframe(raw)
    _render_batch_result(scored)


def _render_batch_result(scored: pd.DataFrame):
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Machines scored", len(scored))
    c2.metric("Critical", int((scored["Status"] == "Critical").sum()))
    c3.metric("Avg Health Score", f"{scored['Health Score'].mean():.1f}")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    def highlight_row(row):
        fg, bg = STATUS_COLORS.get(row["Status"], (TEXT_SECONDARY, "transparent"))
        styles = []
        for col in row.index:
            if col == "Status":
                styles.append(f"background-color:{bg}; color:{fg}; font-weight:700;")
            elif col in ("Risk Score", "Health Score", "Confidence"):
                styles.append(f"color:{fg}; font-weight:600;")
            else:
                styles.append("")
        return styles

    display_cols = RAW_COLUMNS + ["Risk Score", "Health Score", "Confidence", "Status", "Recommendation"]
    styled = (
        scored[display_cols]
        .style
        .apply(highlight_row, axis=1)
        .format({"Risk Score": "{:.1f}", "Health Score": "{:.1f}", "Confidence": "{:.1f}"})
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=440)

    csv_bytes = scored[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        csv_bytes,
        file_name="predictwise_predictions.csv",
        mime="text/csv",
    )