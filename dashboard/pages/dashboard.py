"""
dashboard/pages/dashboard.py

The main fleet Dashboard page. No ML logic lives here - all scoring
comes from dashboard.data.load_scored_fleet(), which itself only
calls src/pipeline.py. This file is UI only: layout, charts, tables.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from dashboard.theme import (
    HEALTHY, WARNING, CRITICAL, BLUE,
    HEALTHY_SOFT, WARNING_SOFT, CRITICAL_SOFT,
    TEXT_PRIMARY, TEXT_SECONDARY,
    STATUS_COLORS, badge_html, kpi_card, base_plotly_layout,
)
from dashboard.data import load_scored_fleet


def show():
    df = load_scored_fleet()

    _render_header()
    _render_ai_summary(df)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    _render_kpis(df)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    _render_charts(df)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _render_table(df)


def _render_header():
    now_str = datetime.now().strftime("%d %b %Y, %H:%M")
    st.markdown(f"""
<div class="header-row">
    <div>
        <div class="main-title">🏭 PredictWise</div>
        <div class="sub-title">Predictive Maintenance Decision Support</div>
    </div>
    <div class="header-right">
        <div class="info-pill">
            <div class="info-pill-label">Dataset</div>
            <div class="info-pill-value">10,000 units</div>
        </div>
        <div class="info-pill">
            <div class="info-pill-label">Model</div>
            <div class="info-pill-value">Random Forest</div>
        </div>
        <div class="info-pill">
            <div class="info-pill-label">Accuracy</div>
            <div class="info-pill-value">98.7%</div>
        </div>
        <div class="live-indicator">
            <div class="live-dot"></div>
            LIVE · {now_str}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def _render_ai_summary(df: pd.DataFrame):
    warning = (df["Status"] == "Warning").sum()
    critical = (df["Status"] == "Critical").sum()
    avg_health = df["Health Score"].mean()
    total = len(df)

    fleet_state = "Fleet operating normally" if critical / total < 0.05 else "Fleet requires elevated attention"

    summary_lines = (
        f"• {fleet_state}<br>"
        f"• {warning:,} machines require attention (Warning)<br>"
        f"• {critical:,} machines need immediate maintenance (Critical)<br>"
        f"• Average Health Score: {avg_health:.1f}"
    )

    st.markdown(
        '<div class="summary-panel"><div class="summary-title">🧠 AI Fleet Summary</div>'
        f'<div class="summary-line">{summary_lines}</div></div>',
        unsafe_allow_html=True
    )


def _render_kpis(df: pd.DataFrame):
    healthy = (df["Status"] == "Healthy").sum()
    warning = (df["Status"] == "Warning").sum()
    critical = (df["Status"] == "Critical").sum()
    avg_health = df["Health Score"].mean()
    std_health = df["Health Score"].std()
    total = len(df)

    col1, col2, col3, col4 = st.columns(4)

    # Comparison lines use only real, computable figures (share of
    # fleet, std dev) rather than invented day-over-day deltas, since
    # the dataset has no time dimension to support a genuine trend.
    kpi_card(col1, "🟢", "Healthy", f"{healthy:,}", HEALTHY, "rgba(34,197,94,0.14)", "Within normal range")
    kpi_card(col2, "🟡", "Warning", f"{warning:,}", WARNING, "rgba(245,158,11,0.14)", f"{warning/total:.1%} of fleet")
    kpi_card(col3, "🔴", "Critical", f"{critical:,}", CRITICAL, "rgba(239,68,68,0.14)", "Needs attention")
    kpi_card(col4, "🔵", "Avg Health Score", f"{avg_health:.1f}", BLUE, "rgba(59,130,246,0.14)", f"σ ± {std_health:.1f}")


def _render_charts(df: pd.DataFrame):
    left, right = st.columns([0.85, 1.15])

    with left:
        _render_donut(df)

    with right:
        _render_priority_panel(df)


def _render_donut(df: pd.DataFrame):
    st.markdown(
        '<div class="chart-card"><div class="chart-title">Machine Status Overview</div>',
        unsafe_allow_html=True
    )

    status_counts = df["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    # softer, desaturated palette + no pull on Healthy so it doesn't
    # visually dominate the chart despite being the largest slice
    color_map = {"Healthy": HEALTHY_SOFT, "Warning": WARNING_SOFT, "Critical": CRITICAL_SOFT}
    pull_map = {"Healthy": 0.0, "Warning": 0.04, "Critical": 0.07}

    fig = px.pie(
        status_counts,
        names="Status",
        values="Count",
        hole=0.68,
        color="Status",
        color_discrete_map=color_map,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        pull=[pull_map.get(s, 0) for s in status_counts["Status"]],
        marker=dict(line=dict(color="#171b22", width=2)),
        sort=False,
    )

    fig.update_layout(
        **base_plotly_layout(height=205, margin=dict(l=10, r=10, t=6, b=30)),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.12,
            xanchor="center", x=0.5,
            font=dict(color=TEXT_PRIMARY, size=12),
            itemwidth=40,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_priority_panel(df: pd.DataFrame):
    st.markdown(
        '<div class="chart-card">'
        '<div class="chart-title">Maintenance Priority Panel</div>'
        '<div class="chart-note">Machines ranked by risk — what to act on first</div>',
        unsafe_allow_html=True
    )

    top_priority = (
        df.sort_values("Risk Score", ascending=False)
          .head(8)
          .reset_index()
    )

    for _, r in top_priority.iterrows():
        fg, _ = STATUS_COLORS.get(r["Status"], (TEXT_SECONDARY, "transparent"))
        row_html = (
            '<div class="priority-row">'
            f'<span class="priority-id">Machine {r["index"]}</span>'
            f'{badge_html(r["Status"])}'
            f'<span class="priority-reco">{r["Recommendation"]}</span>'
            f'<span style="color:{fg}; font-weight:700; font-family:\'IBM Plex Mono\',monospace;">{r["Risk Score"]:.1f}</span>'
            '</div>'
        )
        st.markdown(row_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_table(df: pd.DataFrame):
    st.markdown('<div class="chart-title" style="margin-top:6px;">Machine Overview</div>', unsafe_allow_html=True)

    selected_status = st.radio(
        "Filter Machines",
        ["All", "Healthy", "Warning", "Critical"],
        horizontal=True,
        label_visibility="collapsed",
    )

    filtered_df = df if selected_status == "All" else df[df["Status"] == selected_status]
    filtered_df = filtered_df.sort_values("Risk Score", ascending=False)

    table_cols = [
        "Type", "Air temperature", "Process temperature", "Rotational speed",
        "Torque", "Tool wear", "Risk Score", "Health Score", "Status", "Recommendation"
    ]

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

    styled_table = (
        filtered_df[table_cols]
        .style
        .apply(highlight_row, axis=1)
        .format({"Risk Score": "{:.1f}", "Health Score": "{:.1f}"})
    )

    st.dataframe(styled_table, use_container_width=True, hide_index=True, height=440)

    st.markdown(
        f"<div class='chart-note' style='margin-top:8px;'>{len(filtered_df):,} machines shown.</div>",
        unsafe_allow_html=True
    )