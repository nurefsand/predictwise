"""
dashboard/pages/analytics.py

Fleet analytics: understanding the dataset and overall fleet health.
No predictions happen here - this page only reads the already-scored
dataframe from dashboard.data.load_scored_fleet() (which itself only
calls src/pipeline.py) and visualizes it. No preprocessing is
duplicated here.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.theme import (
    HEALTHY, WARNING, CRITICAL, BLUE,
    HEALTHY_SOFT, WARNING_SOFT, CRITICAL_SOFT,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER,
    STATUS_COLORS, page_title, kpi_card, base_plotly_layout,
)
from dashboard.data import load_scored_fleet

NUMERIC_COLS = [
    "Air temperature", "Process temperature", "Rotational speed",
    "Torque", "Tool wear", "Risk Score", "Health Score",
]


def show():
    df = load_scored_fleet()

    page_title("📊", "Analytics", "Fleet statistics and historical analysis")

    _render_kpis(df)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_distribution_charts(df)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_score_histograms(df)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_correlation_heatmap(df)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_failure_analysis(df)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_data_table(df)


# ---------------------------------------------------------------
# SECTION 1 - KPI CARDS
# ---------------------------------------------------------------

def _render_kpis(df: pd.DataFrame):
    total = len(df)
    healthy = (df["Status"] == "Healthy").sum()
    warning = (df["Status"] == "Warning").sum()
    critical = (df["Status"] == "Critical").sum()
    avg_health = df["Health Score"].mean()
    avg_risk = df["Risk Score"].mean()

    row1 = st.columns(3)
    kpi_card(row1[0], "🏭", "Total Machines", f"{total:,}", BLUE, "rgba(59,130,246,0.14)", "Fleet size")
    kpi_card(row1[1], "🟢", "Healthy", f"{healthy:,}", HEALTHY, "rgba(34,197,94,0.14)", f"{healthy/total:.1%} of fleet")
    kpi_card(row1[2], "🟡", "Warning", f"{warning:,}", WARNING, "rgba(245,158,11,0.14)", f"{warning/total:.1%} of fleet")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    row2 = st.columns(3)
    kpi_card(row2[0], "🔴", "Critical", f"{critical:,}", CRITICAL, "rgba(239,68,68,0.14)", f"{critical/total:.1%} of fleet")
    kpi_card(row2[1], "💙", "Avg Health Score", f"{avg_health:.1f}", BLUE, "rgba(59,130,246,0.14)", "Higher is better")
    kpi_card(row2[2], "⚠️", "Avg Risk Score", f"{avg_risk:.1f}", WARNING, "rgba(245,158,11,0.14)", "Lower is better")


# ---------------------------------------------------------------
# SECTION 2 - STATUS + TYPE DISTRIBUTION
# ---------------------------------------------------------------

def _render_distribution_charts(df: pd.DataFrame):
    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="chart-card"><div class="chart-title">Machine Status Distribution</div>',
            unsafe_allow_html=True
        )

        status_counts = df["Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        color_map = {"Healthy": HEALTHY_SOFT, "Warning": WARNING_SOFT, "Critical": CRITICAL_SOFT}

        fig = px.pie(
            status_counts, names="Status", values="Count",
            hole=0.65, color="Status", color_discrete_map=color_map,
        )
        fig.update_traces(
            textposition="inside", textinfo="percent",
            marker=dict(line=dict(color="#171b22", width=2)),
            sort=False,
        )
        fig.update_layout(
            **base_plotly_layout(height=260, margin=dict(l=10, r=10, t=6, b=30)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5,
                        font=dict(color=TEXT_PRIMARY, size=12)),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="chart-card"><div class="chart-title">Machine Type Distribution</div>',
            unsafe_allow_html=True
        )

        type_counts = df["Type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]

        fig = px.bar(
            type_counts, x="Type", y="Count", text="Count",
            color="Type", color_discrete_sequence=[BLUE, HEALTHY_SOFT, WARNING_SOFT],
        )
        fig.update_traces(textposition="outside", textfont=dict(color=TEXT_PRIMARY, size=11))
        fig.update_layout(
            **base_plotly_layout(height=260),
            showlegend=False,
            xaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER),
            yaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 3 - SCORE HISTOGRAMS
# ---------------------------------------------------------------

def _render_score_histograms(df: pd.DataFrame):
    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="chart-card"><div class="chart-title">Health Score Distribution</div>',
            unsafe_allow_html=True
        )
        fig = px.histogram(df, x="Health Score", nbins=30, color_discrete_sequence=[HEALTHY_SOFT])
        fig.update_layout(
            **base_plotly_layout(height=240),
            xaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER),
            yaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER, title="Machines"),
            bargap=0.05,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="chart-card"><div class="chart-title">Risk Score Distribution</div>',
            unsafe_allow_html=True
        )
        fig = px.histogram(df, x="Risk Score", nbins=30, color_discrete_sequence=[CRITICAL_SOFT])
        fig.update_layout(
            **base_plotly_layout(height=240),
            xaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER),
            yaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER, title="Machines"),
            bargap=0.05,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 4 - CORRELATION HEATMAP
# ---------------------------------------------------------------

def _render_correlation_heatmap(df: pd.DataFrame):
    st.markdown(
        '<div class="chart-card"><div class="chart-title">Correlation Heatmap</div>'
        '<div class="chart-note">Relationship between sensor readings and computed scores</div>',
        unsafe_allow_html=True
    )

    corr = df[NUMERIC_COLS].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=[[0, CRITICAL_SOFT], [0.5, "#171b22"], [1, BLUE]],
        zmin=-1, zmax=1,
        aspect="auto",
    )
    fig.update_traces(textfont=dict(size=11))
    fig.update_layout(
        **base_plotly_layout(height=420, margin=dict(l=10, r=10, t=6, b=6)),
        xaxis=dict(color=TEXT_SECONDARY, side="bottom"),
        yaxis=dict(color=TEXT_SECONDARY),
        coloraxis_colorbar=dict(tickfont=dict(color=TEXT_SECONDARY)),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 5 - FAILURE ANALYSIS
# ---------------------------------------------------------------

def _render_failure_analysis(df: pd.DataFrame):
    st.markdown(
        '<div class="chart-card"><div class="chart-title">Failure Analysis</div>'
        '<div class="chart-note">Average Risk Score by Machine Type</div>',
        unsafe_allow_html=True
    )

    by_type = df.groupby("Type")["Risk Score"].mean().reset_index().sort_values("Risk Score", ascending=False)

    fig = px.bar(
        by_type, x="Type", y="Risk Score", text="Risk Score",
        color="Risk Score", color_continuous_scale=[[0, WARNING_SOFT], [1, CRITICAL]],
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                       textfont=dict(color=TEXT_PRIMARY, size=11))
    fig.update_layout(
        **base_plotly_layout(height=280),
        coloraxis_showscale=False,
        xaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER),
        yaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 6 - INTERACTIVE DATA TABLE
# ---------------------------------------------------------------

def _render_data_table(df: pd.DataFrame):
    st.markdown('<div class="chart-title">Fleet Data</div>', unsafe_allow_html=True)

    selected_status = st.radio(
        "Filter by Status",
        ["All", "Healthy", "Warning", "Critical"],
        horizontal=True,
        label_visibility="collapsed",
        key="analytics_status_filter",
    )

    filtered_df = df if selected_status == "All" else df[df["Status"] == selected_status]

    def highlight_row(row):
        fg, bg = STATUS_COLORS.get(row["Status"], (TEXT_SECONDARY, "transparent"))
        styles = []
        for col in row.index:
            if col == "Status":
                styles.append(f"background-color:{bg}; color:{fg}; font-weight:700;")
            elif col in ("Risk Score", "Health Score"):
                styles.append(f"color:{fg}; font-weight:700;")
            else:
                styles.append("")
        return styles

    display_cols = [
        "Type", "Air temperature", "Process temperature", "Rotational speed",
        "Torque", "Tool wear", "Risk Score", "Health Score", "Status", "Recommendation"
    ]

    styled = (
        filtered_df[display_cols]
        .style
        .apply(highlight_row, axis=1)
        .format({"Risk Score": "{:.1f}", "Health Score": "{:.1f}"})
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=440)

    st.markdown(
        f"<div class='chart-note' style='margin-top:8px;'>{len(filtered_df):,} machines shown.</div>",
        unsafe_allow_html=True
    )