"""
dashboard/theme.py

Everything visual that more than one page needs lives here:
color tokens, the global CSS injection, and small reusable render
helpers (kpi_card, badge_html, page_title, base_plotly_layout).

Import from this module in every page instead of redefining colors
or copy-pasting <style> blocks. If you need a new shared component,
add it here once rather than repeating it per page.
"""

import streamlit as st

# =========================================================
# DESIGN TOKENS
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

STATUS_COLORS = {
    "Healthy": (HEALTHY, "rgba(34,197,94,0.14)"),
    "Warning": (WARNING, "rgba(245,158,11,0.14)"),
    "Critical": (CRITICAL, "rgba(239,68,68,0.14)"),
}


def status_color(status: str):
    """Returns (foreground, background) hex/rgba pair for a status."""
    return STATUS_COLORS.get(status, (TEXT_SECONDARY, "transparent"))


def badge_html(status: str) -> str:
    """A small colored pill for Status values. Single-line output on
    purpose - multi-line indented HTML passed to st.markdown can get
    misread as a code block and render as literal text."""
    fg, bg = status_color(status)
    return f'<span class="badge" style="color:{fg};background:{bg};">{status}</span>'


def kpi_card(col, icon: str, label: str, value: str, color: str, color_soft: str, compare: str = ""):
    """Renders one KPI card into a given st.columns() slot."""
    with col:
        html = (
            f'<div class="kpi-card" style="--kpi-color:{color}; --kpi-color-soft:{color_soft};">'
            f'<div class="kpi-top-row"><div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-label">{label}</div></div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-compare">{compare}</div></div>'
        )
        st.markdown(html, unsafe_allow_html=True)


def page_title(icon: str, title: str, subtitle: str = ""):
    """Compact page header used by every page except Dashboard
    (Dashboard keeps its own fuller fleet-status header)."""
    sub = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="page-title">{icon} {title}</div>{sub}',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


def base_plotly_layout(height: int = 260, margin: dict | None = None) -> dict:
    """
    Common Plotly layout kwargs shared by every chart.

    Usage:
        fig.update_layout(**base_plotly_layout(height=220), legend=dict(...))

    Do NOT also pass height= or margin= again in the same
    update_layout() call - that's what caused the "multiple values
    for keyword argument 'margin'" error earlier. If a chart needs a
    different margin, pass it to this function instead:
        base_plotly_layout(height=220, margin=dict(l=10, r=10, t=6, b=30))
    """
    return dict(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT_SECONDARY, family="IBM Plex Mono", size=11),
        height=height,
        margin=margin or dict(l=10, r=10, t=6, b=6),
    )


def inject_theme():
    """Injects the global CSS once. Call this a single time in
    dashboard/app.py, before routing to any page - not inside every
    page's show(), or the stylesheet gets duplicated on every rerun."""
    st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

/* Hide Streamlit's own menu/deploy clutter, but NOT the whole header -
   the whole header contains the "reopen sidebar" arrow
   ([data-testid="stExpandSidebarButton"]), and hiding the header
   hides that button too, permanently trapping the user with a
   collapsed sidebar and no way to reopen it. Target only the
   specific elements we don't want instead. */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
[data-testid="stMainMenu"] {{visibility: hidden;}}
[data-testid="stAppDeployButton"] {{visibility: hidden;}}
[data-testid="stHeader"] {{
    background: transparent;
    box-shadow: none;
}}

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

/* ---------------- SIDEBAR NAV ---------------- */

section[data-testid="stSidebar"] {{
    background-color: {CARD};
    border-right: 1px solid {BORDER};
}}

.sidebar-logo {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    padding: 6px 0 16px 0;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] {{
    flex-direction: column;
    gap: 4px;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    width: 100%;
    border-radius: 8px;
}}

/* ---------------- GENERIC PAGE HEADER ---------------- */

.page-title {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 30px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    letter-spacing: 0.4px;
}}

.page-subtitle {{
    color: {TEXT_SECONDARY};
    font-size: 12.5px;
    opacity: 0.85;
    margin-top: 2px;
}}

/* ---------------- HEADER (Dashboard page) ---------------- */

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

.live-dot-static{{
    background: {TEXT_SECONDARY};
    box-shadow: none;
    animation: none;
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
    min-height: 84px;
    margin-bottom: 20px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.14);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}}

.kpi-card:hover{{
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.22);
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
    font-size: 11px;
    color: var(--kpi-color, {TEXT_SECONDARY});
    margin-top: 5px;
    font-family: 'IBM Plex Mono', monospace;
}}

/* ---------- Unified Section Card ----------
   Every bordered "section" on every page (a title + optional subtitle
   + a chart/table/widgets below it) renders through ONE mechanism:
   dashboard.layout.section(), which wraps content in
   st.container(border=True, key="section_<slug>"). Targeting the key
   prefix here means every section on every page gets IDENTICAL
   background/border/radius/padding/shadow - one card style, not a
   per-page reimplementation of it. */

[class*="st-key-section_"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    padding: 16px 20px 18px 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.14) !important;
}}

.section-title{{
    font-family: 'Rajdhani', sans-serif;
    font-size: 15.5px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    letter-spacing: 0.5px;
    margin-bottom: 2px;
    text-transform: uppercase;
}}

.section-subtitle {{
    font-size: 11px;
    color: {TEXT_SECONDARY};
    margin-bottom: 10px;
}}

/* ---------- Responsive equal-height card grid ----------
   Used by dashboard.layout.card_grid() for any "N cards per row"
   layout (About's tech/feature cards, and any future page that needs
   the same thing). CSS Grid's default row-stretch behavior is what
   makes every card in the same row exactly equal height - no manual
   pixel heights anywhere. */

.card-grid {{
    display: grid;
    grid-template-columns: repeat(var(--grid-cols, 4), 1fr);
    gap: 14px;
}}

@media (max-width: 1100px) {{
    .card-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}

@media (max-width: 640px) {{
    .card-grid {{ grid-template-columns: 1fr; }}
}}

.grid-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 18px;
    height: 100%;
    box-sizing: border-box;
}}

.grid-card-icon {{
    font-size: 22px;
    margin-bottom: 8px;
}}

.grid-card-title {{
    font-weight: 700;
    font-size: 14.5px;
    color: {TEXT_PRIMARY};
    margin-bottom: 4px;
}}

.grid-card-desc {{
    font-size: 12.5px;
    color: {TEXT_SECONDARY};
    line-height: 1.5;
}}

/* ---------- Workflow strip (arrows between steps) ---------- */

.workflow-strip {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
}}

.workflow-step {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    min-width: 140px;
    flex: 1 1 140px;
}}

.workflow-icon {{
    font-size: 20px;
    margin-bottom: 6px;
}}

.workflow-label {{
    font-size: 12.5px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}

.workflow-arrow {{
    font-size: 18px;
    color: {TEXT_SECONDARY};
    flex: 0 0 auto;
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

/* ---------------- FILTER / SEGMENTED PILLS ---------------- */

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