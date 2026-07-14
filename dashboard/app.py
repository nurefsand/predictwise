import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
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

# =========================================================
# THEME TOKENS
# Cyber-industrial: Siemens / ABB / Azure IoT style dark UI.
# =========================================================

BG = "#0f1117"
CARD = "#171b22"
BORDER = "#2b313d"
TEXT_PRIMARY = "#e8edf2"
TEXT_SECONDARY = "#7d8a97"
HEALTHY = "#22c55e"
WARNING = "#f59e0b"
CRITICAL = "#ef4444"
BLUE = "#3b82f6"

# softer variants used only where "healthy" shouldn't visually dominate
HEALTHY_SOFT = "#86efac"
WARNING_SOFT = "#fcd34d"
CRITICAL_SOFT = "#fca5a5"

st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

.stApp {{
    background-color: {BG};
    background-image:
        radial-gradient(ellipse 900px 400px at 50% -10%, rgba(59,130,246,0.05), transparent 70%),
        linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px);
    background-size: auto, 46px 46px, 46px 46px;
}}

.block-container{{
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
    max-width: 1440px;
}}

/* ---------------- HEADER ---------------- */

.header-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}}

.main-title{{
    font-family: 'Rajdhani', sans-serif;
    font-size: 37px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    letter-spacing: 0.4px;
    margin-bottom: 1px;
    line-height: 1.1;
}}

.sub-title{{
    color: {TEXT_SECONDARY};
    font-size: 11.5px;
    font-weight: 400;
    letter-spacing: 0.3px;
    opacity: 0.85;
}}

.header-right {{
    display: flex;
    align-items: center;
    gap: 10px;
}}

.info-pill {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 14px;
    text-align: left;
    min-width: 92px;
}}

.info-pill-label{{
    font-size: 10px;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}

.info-pill-value{{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13.5px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin-top: 1px;
}}

.live-indicator{{
    display: flex;
    align-items: center;
    gap: 7px;
    color: {TEXT_SECONDARY};
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    letter-spacing: 0.8px;
    padding-left: 6px;
}}

.live-dot{{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: {HEALTHY};
    box-shadow: 0 0 6px {HEALTHY};
    animation: pulse 1.8s infinite;
}}

@keyframes pulse {{
    0% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
    100% {{ opacity: 1; }}
}}

/* ---------------- KPI CARDS ---------------- */

.kpi-card{{
    background: {CARD};
    border-radius: 10px;
    border-left: 3px solid var(--kpi-color, {BLUE});
    padding: 14px 16px;
    height: 100px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}}

.kpi-card:hover{{
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.28);
}}

.kpi-top-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}}

.kpi-icon {{
    width: 22px;
    height: 22px;
    border-radius: 6px;
    background: var(--kpi-color-soft, rgba(59,130,246,0.12));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
}}

.kpi-label{{
    font-size: 11.5px;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}

.kpi-value{{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1;
}}

.kpi-compare{{
    font-size:13px;
    font-weight:500;
    color:var(--kpi-color,{TEXT_SECONDARY});
    margin-top:8px;
    font-family:'IBM Plex Mono', monospace;
}}

/* ---------- Chart Cards ---------- */

.chart-card{{
    background: {CARD};
    border-radius: 10px;
    padding: 18px 20px;
    height: 100%;
}}

.chart-title{{
    font-family: 'Rajdhani', sans-serif;
    font-size: 15.5px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    letter-spacing: 0.5px;
    margin-bottom: 4px;
    text-transform: uppercase;
}}

.chart-note {{
    font-size: 11px;
    color: {TEXT_SECONDARY};
    margin-bottom: 10px;
}}

/* Streamlit native tweaks */
div[data-baseweb="select"] > div {{
    background-color: {CARD};
    border-color: {BORDER};
    border-radius: 8px;
}}

label, .stSelectbox label {{
    color: {TEXT_SECONDARY} !important;
    font-size: 12px !important;
}}

hr {{
    border-color: {BORDER} !important;
    margin-top: 18px;
    margin-bottom: 18px;
}}

/* ---------------- AI SUMMARY PANEL ---------------- */

.summary-panel {{
    background: linear-gradient(180deg, #182029 0%, {CARD} 100%);
    border-radius: 10px;
    border-left: 3px solid {BLUE};
    padding: 16px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.18);
}}

.summary-title {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}}

.summary-line {{
    font-size: 13px;
    color: {TEXT_PRIMARY};
    line-height: 1.9;
}}

/* ---------------- FILTER PILLS ---------------- */

div[role="radiogroup"] {{
    gap: 6px;
}}

div[role="radiogroup"] label {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 999px;
    padding: 5px 16px !important;
    margin: 0 !important;
    transition: all 0.15s ease;
}}

div[role="radiogroup"] label:hover {{
    border-color: {BLUE};
}}

div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {{
    font-size: 12.5px !important;
    color: {TEXT_SECONDARY};
}}

/* ---------------- MAINTENANCE PRIORITY PANEL ---------------- */

.priority-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 9px 4px;
    border-bottom: 1px solid {BORDER};
    font-size: 12.5px;
}}

.priority-row:last-child {{
    border-bottom: none;
}}

.priority-id {{
    font-family: 'IBM Plex Mono', monospace;
    color: {TEXT_PRIMARY};
    width: 90px;
    flex-shrink: 0;
}}

.priority-reco {{
    color: {TEXT_SECONDARY};
    flex-grow: 1;
    padding: 0 12px;
}}

.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 10.5px;
    font-weight: 600;
    white-space: nowrap;
}}

</style>
""", unsafe_allow_html=True)

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
df["Recommendation"] = df["Risk Score"].apply(maintenance_recommendation)

# -------------------------
# HEADER
# -------------------------

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

# -------------------------
# AI FLEET SUMMARY (main insight panel)
# -------------------------

healthy = (df["Status"] == "Healthy").sum()
warning = (df["Status"] == "Warning").sum()
critical = (df["Status"] == "Critical").sum()
avg_health = df["Health Score"].mean()
std_health = df["Health Score"].std()
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

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# -------------------------
# KPI
# -------------------------

col1, col2, col3, col4 = st.columns(4)

# Comparison lines use only real, computable figures (share of fleet,
# std dev) rather than invented day-over-day deltas, since the dataset
# has no time dimension to support a genuine trend claim.
kpi_data = [
    (col1, "🟢", "Healthy", f"{healthy:,}", HEALTHY, "rgba(34,197,94,0.14)", "Within normal range"),
    (col2, "🟡", "Warning", f"{warning:,}", WARNING, "rgba(245,158,11,0.14)", f"{warning/total:.1%} of fleet"),
    (col3, "🔴", "Critical", f"{critical:,}", CRITICAL, "rgba(239,68,68,0.14)", "Needs attention"),
    (col4, "🔵", "Avg Health Score", f"{avg_health:.1f}", BLUE, "rgba(59,130,246,0.14)", f"σ ± {std_health:.1f}"),
]

for col, icon, label, value, color, color_soft, compare in kpi_data:
    with col:
        kpi_html = (
            f'<div class="kpi-card" style="--kpi-color:{color}; --kpi-color-soft:{color_soft};">'
            f'<div class="kpi-top-row"><div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-label">{label}</div></div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-compare">{compare}</div></div>'
        )
        st.markdown(kpi_html, unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# -------------------------
# CHARTS
# -------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(color=TEXT_SECONDARY, family="IBM Plex Mono", size=11),
    margin=dict(l=10, r=10, t=6, b=6),
)

left, right = st.columns([0.85, 1.15])

with left:
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
        marker=dict(line=dict(color=CARD, width=2)),
        sort=False,
    )

    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT_SECONDARY, family="IBM Plex Mono", size=11),
        height=350,
        margin=dict(l=10, r=10, t=6, b=30),
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

with right:
    st.markdown(
        '<div class="chart-card">'
        '<div class="chart-title">Maintenance Priority Panel</div>'
        '<div class="chart-note">Machines ranked by risk — what to act on first</div>',
        unsafe_allow_html=True
    )

    status_colors_local = {
        "Healthy": (HEALTHY, "rgba(34,197,94,0.14)"),
        "Warning": (WARNING, "rgba(245,158,11,0.14)"),
        "Critical": (CRITICAL, "rgba(239,68,68,0.14)"),
    }

    top_priority = (
        df.sort_values("Risk Score", ascending=False)
          .head(8)
          .reset_index()
    )

    for _, r in top_priority.iterrows():
        fg, bg = status_colors_local.get(r["Status"], (TEXT_SECONDARY, "transparent"))
        badge_html = f'<span class="badge" style="color:{fg};background:{bg};">{r["Status"]}</span>'
        row_html = (
            '<div class="priority-row">'
            f'<span class="priority-id">Machine {r["index"]}</span>'
            f'{badge_html}'
            f'<span class="priority-reco">{r["Recommendation"]}</span>'
            f'<span style="color:{fg}; font-weight:700; font-family:\'IBM Plex Mono\',monospace;">{r["Risk Score"]:.1f}</span>'
            '</div>'
        )
        st.markdown(row_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# -------------------------
# FILTER
# -------------------------

selected_status = st.radio(
    "Filter Machines",
    ["All", "Healthy", "Warning", "Critical"],
    horizontal=True,
    label_visibility="collapsed",
)

filtered_df = df if selected_status == "All" else df[df["Status"] == selected_status]
filtered_df = filtered_df.sort_values("Risk Score", ascending=False)

# -------------------------
# TABLE
# -------------------------

st.markdown('<div class="chart-title" style="margin-top:6px;">Machine Overview</div>', unsafe_allow_html=True)

status_colors = {
    "Healthy": (HEALTHY, "rgba(34,197,94,0.14)"),
    "Warning": (WARNING, "rgba(245,158,11,0.14)"),
    "Critical": (CRITICAL, "rgba(239,68,68,0.14)"),
}

table_cols = [
    "Type", "Air temperature", "Process temperature", "Rotational speed",
    "Torque", "Tool wear", "Risk Score", "Health Score", "Status", "Recommendation"
]

def highlight_row(row):
    fg, bg = status_colors.get(row["Status"], (TEXT_SECONDARY, "transparent"))
    styles = []
    for col in row.index:
        if col == "Status":
            styles.append(f"background-color:{bg}; color:{fg}; font-weight:700; border-radius:4px;")
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

# Native st.dataframe: built-in column sorting (click header),
# scrolling, and the column search/filter toolbar — no custom HTML.
st.dataframe(
    styled_table,
    use_container_width=True,
    hide_index=True,
    height=440,
)

st.markdown(
    f"<div class='chart-note' style='margin-top:8px;'>{len(filtered_df):,} machines shown.</div>",
    unsafe_allow_html=True
)