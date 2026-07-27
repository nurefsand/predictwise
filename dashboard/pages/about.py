"""
dashboard/pages/about.py

Project introduction page. Mostly static/descriptive content, but
every number in Section 7 (Project Statistics) is read live from the
dataset, the feature pipeline, the trained model, or src/business_rules.py
- nothing there is a hardcoded/invented figure.

Note on markup style: every section below is built as ONE self-contained
st.markdown(..., unsafe_allow_html=True) call (opening and closing tags
in the same call). This project's other pages ran into a real bug where
splitting a card's opening <div> and closing </div> across two separate
st.markdown calls silently failed to visually enclose native widgets in
between (Streamlit does not treat separate markdown calls as one
continuous HTML stream). Keeping each card fully self-contained in one
call avoids that entirely.
"""

from pathlib import Path

import streamlit as st
import pandas as pd

from dashboard.theme import (
    HEALTHY, WARNING, CRITICAL, BLUE, TEXT_PRIMARY, TEXT_SECONDARY, BORDER, CARD,
    kpi_card,
)
from dashboard.data import load_scored_fleet
from src.pipeline import get_feature_matrix
from src.business_rules import calculate_status

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "predictive_maintenance.csv"

CANDIDATE_LABEL_COLUMNS = [
    "Machine failure", "machine failure", "Machine Failure", "Target", "target",
]

_ABOUT_CSS = """
<style>
.about-hero-title{
    font-family: 'Rajdhani', sans-serif;
    font-size: 40px;
    font-weight: 700;
    color: """ + TEXT_PRIMARY + """;
    margin-bottom: 2px;
}
.about-hero-subtitle{
    font-size: 15px;
    color: """ + TEXT_SECONDARY + """;
    margin-bottom: 14px;
}
.about-hero-intro{
    font-size: 13.5px;
    color: """ + TEXT_PRIMARY + """;
    line-height: 1.8;
    max-width: 900px;
}
.about-grid{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
}
.about-tech-card, .about-feature-card{
    background: """ + CARD + """;
    border: 1px solid """ + BORDER + """;
    border-radius: 10px;
    padding: 16px 18px;
    flex: 1 1 220px;
    min-width: 220px;
}
.about-tech-icon, .about-feature-icon{
    font-size: 22px;
    margin-bottom: 8px;
}
.about-tech-name, .about-feature-title{
    font-weight: 700;
    font-size: 14.5px;
    color: """ + TEXT_PRIMARY + """;
    margin-bottom: 4px;
}
.about-tech-desc, .about-feature-desc{
    font-size: 12.5px;
    color: """ + TEXT_SECONDARY + """;
    line-height: 1.5;
}
.about-workflow{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
}
.about-workflow-step{
    background: """ + CARD + """;
    border: 1px solid """ + BORDER + """;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    min-width: 140px;
    flex: 1 1 140px;
}
.about-workflow-icon{
    font-size: 20px;
    margin-bottom: 6px;
}
.about-workflow-label{
    font-size: 12.5px;
    font-weight: 600;
    color: """ + TEXT_PRIMARY + """;
}
.about-workflow-arrow{
    font-size: 18px;
    color: """ + TEXT_SECONDARY + """;
    flex: 0 0 auto;
}
.about-arch-group{
    margin-bottom: 14px;
}
.about-arch-path{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: """ + BLUE + """;
    font-weight: 600;
    margin-bottom: 6px;
}
.about-arch-item{
    font-size: 12.5px;
    color: """ + TEXT_SECONDARY + """;
    padding: 3px 0 3px 18px;
    border-left: 2px solid """ + BORDER + """;
    margin-left: 4px;
}
.about-footer{
    text-align: center;
    padding: 26px 20px;
}
.about-footer-title{
    font-family: 'Rajdhani', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: """ + TEXT_PRIMARY + """;
}
.about-footer-sub{
    font-size: 13px;
    color: """ + TEXT_SECONDARY + """;
    margin-top: 4px;
}
.about-footer-tag{
    font-size: 12px;
    color: """ + TEXT_SECONDARY + """;
    margin-top: 10px;
    font-style: italic;
}
</style>
"""


def show():
    st.markdown(_ABOUT_CSS, unsafe_allow_html=True)

    _render_hero()
    _spacer()
    _render_overview()
    _spacer()
    _render_workflow()
    _spacer()
    _render_architecture()
    _spacer()
    _render_tech_stack()
    _spacer()
    _render_features()
    _spacer()
    _render_statistics()
    _spacer()
    _render_footer()


def _spacer():
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 1 - HERO
# ---------------------------------------------------------------

def _render_hero():
    st.markdown(
        '<div class="about-hero-title">🏭 About PredictWise</div>'
        '<div class="about-hero-subtitle">AI-powered predictive maintenance decision '
        'support system for industrial equipment.</div>'
        '<div class="about-hero-intro">PredictWise analyzes industrial machine sensor '
        'data — temperature, rotational speed, torque, and tool wear — using a trained '
        'Random Forest model to estimate maintenance risk, translate that risk into a '
        'health score, and surface a concrete maintenance recommendation for each '
        'machine.</div>',
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------
# SECTION 2 - PROJECT OVERVIEW
# ---------------------------------------------------------------

def _render_overview():
    with st.container(border=True, key="about_card_overview"):
        st.markdown(
            '<div class="chart-title">Project Overview</div>'
            '<div class="summary-line" style="margin-top:8px;">'
            "Unplanned downtime is one of the most expensive problems on a factory floor - "
            "a machine that fails without warning stops production, not just itself. "
            "Predictive maintenance flips this around: instead of reacting after a "
            "breakdown, or servicing machines on a fixed schedule regardless of their "
            "actual condition, it uses live sensor readings to estimate how close a "
            "machine is to failing right now.<br><br>"
            "PredictWise puts that idea into a working tool. It takes the same sensor "
            "signals a machine already produces, runs them through a trained model, and "
            "turns the output into something a maintenance engineer can act on "
            "immediately: which machines are healthy, which need a closer look, and "
            "which need attention now - before they fail. The goal is straightforward: "
            "fewer surprise breakdowns, less unnecessary maintenance, and maintenance "
            "decisions grounded in actual machine condition instead of guesswork."
            '</div>',
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------
# SECTION 3 - WORKFLOW
# ---------------------------------------------------------------

def _render_workflow():
    steps = [
        ("📡", "Sensor Data"),
        ("🛠️", "Feature Engineering"),
        ("🌲", "Random Forest Prediction"),
        ("📐", "Business Rules"),
        ("⚠️", "Risk Score"),
        ("✅", "Maintenance Recommendation"),
    ]

    with st.container(border=True, key="about_card_workflow"):
        st.markdown('<div class="chart-title">How PredictWise Works</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        parts = []
        for i, (icon, label) in enumerate(steps):
            parts.append(
                f'<div class="about-workflow-step">'
                f'<div class="about-workflow-icon">{icon}</div>'
                f'<div class="about-workflow-label">{label}</div></div>'
            )
            if i < len(steps) - 1:
                parts.append('<div class="about-workflow-arrow">→</div>')

        st.markdown(f'<div class="about-workflow">{"".join(parts)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 4 - PROJECT ARCHITECTURE
# ---------------------------------------------------------------

def _render_architecture():
    groups = [
        ("dashboard/", ["Streamlit application (routing, theme, cached data access)"]),
        ("dashboard/pages/", [
            "Dashboard", "AI Prediction", "Analytics", "Explainability",
            "Model Performance", "Settings", "About",
        ]),
        ("src/", [
            "Prediction engine", "Feature engineering", "Preprocessing",
            "Business rules", "Explainability",
        ]),
        ("models/", ["Trained Random Forest (random_forest.pkl)"]),
        ("data/", ["AI4I 2020 predictive maintenance dataset"]),
    ]

    with st.container(border=True, key="about_card_architecture"):
        st.markdown('<div class="chart-title">Project Architecture</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        html = ""
        for path, items in groups:
            items_html = "".join(f'<div class="about-arch-item">{item}</div>' for item in items)
            html += f'<div class="about-arch-group"><div class="about-arch-path">{path}</div>{items_html}</div>'

        st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 5 - TECHNOLOGY STACK
# ---------------------------------------------------------------

def _render_tech_stack():
    stack = [
        ("🐍", "Python", "Core language for the whole application and ML pipeline"),
        ("🎈", "Streamlit", "Powers the interactive web dashboard"),
        ("🌲", "Scikit-learn", "Trains and runs the Random Forest model"),
        ("🐼", "Pandas", "Data loading, transformation, and feature tables"),
        ("🔢", "NumPy", "Numerical operations underlying the pipeline"),
        ("📊", "Plotly", "All interactive charts across the app"),
        ("🧠", "SHAP", "Explains individual and global model predictions"),
        ("🌳", "Random Forest", "The predictive model itself"),
    ]

    with st.container(border=True, key="about_card_tech"):
        st.markdown('<div class="chart-title">Technology Stack</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        cards = "".join(
            f'<div class="about-tech-card">'
            f'<div class="about-tech-icon">{icon}</div>'
            f'<div class="about-tech-name">{name}</div>'
            f'<div class="about-tech-desc">{desc}</div></div>'
            for icon, name, desc in stack
        )
        st.markdown(f'<div class="about-grid">{cards}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 6 - APPLICATION FEATURES
# ---------------------------------------------------------------

def _render_features():
    features = [
        ("📊", "Dashboard", "Live fleet overview with AI summary, KPIs, status breakdown, and maintenance priorities."),
        ("🔮", "AI Prediction", "Scores a single machine or a batch CSV upload for risk, health, and recommendation."),
        ("📈", "Analytics", "Fleet-wide statistics: distributions, correlations, and risk by machine type."),
        ("🧠", "Explainability", "SHAP-based global and local explanations for why the model predicts what it predicts."),
        ("📉", "Model Performance", "Accuracy, precision/recall, ROC curve, confusion matrix, and feature importance."),
        ("⚙️", "Settings", "Session preferences, live model info, risk thresholds, and system status."),
    ]

    with st.container(border=True, key="about_card_features"):
        st.markdown('<div class="chart-title">Application Features</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        cards = "".join(
            f'<div class="about-feature-card">'
            f'<div class="about-feature-icon">{icon}</div>'
            f'<div class="about-feature-title">{title}</div>'
            f'<div class="about-feature-desc">{desc}</div></div>'
            for icon, title, desc in features
        )
        st.markdown(f'<div class="about-grid">{cards}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 7 - PROJECT STATISTICS (all real, computed live)
# ---------------------------------------------------------------

def _count_prediction_categories() -> int:
    """Reuses the real calculate_status() function (same approach as
    Settings' Risk Thresholds) instead of assuming there are 3 categories."""
    statuses = {calculate_status(s) for s in range(0, 101)}
    return len(statuses)


def _try_real_accuracy():
    """Only returns a value if a real ground-truth column exists in the
    raw CSV. Never fabricates an accuracy figure."""
    if not DATA_PATH.exists():
        return None
    raw = pd.read_csv(DATA_PATH)
    label_col = next((c for c in CANDIDATE_LABEL_COLUMNS if c in raw.columns), None)
    if label_col is None:
        return None
    try:
        from sklearn.metrics import accuracy_score
        df = load_scored_fleet().reset_index(drop=True)
        y_true = raw[label_col].astype(int).reset_index(drop=True)
        if len(y_true) != len(df):
            return None
        y_pred = (df["Status"] != "Healthy").astype(int)
        return accuracy_score(y_true, y_pred)
    except Exception:
        return None


def _render_statistics():
    df = load_scored_fleet()

    try:
        X = get_feature_matrix(df.head(3))
        n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
    except Exception:
        n_features = None

    n_categories = _count_prediction_categories()
    accuracy = _try_real_accuracy()

    with st.container(border=True, key="about_card_stats"):
        st.markdown('<div class="chart-title">Project Statistics</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        kpi_card(c1, "🏭", "Dataset Size", f"{len(df):,}", BLUE, "rgba(59,130,246,0.14)", "machines")
        kpi_card(c2, "🌲", "Model Type", "Random Forest", HEALTHY, "rgba(34,197,94,0.14)")
        kpi_card(c3, "🔢", "Features Used",
                  str(n_features) if n_features is not None else "N/A",
                  WARNING, "rgba(245,158,11,0.14)")
        kpi_card(c4, "🎯", "Prediction Categories", str(n_categories), CRITICAL, "rgba(239,68,68,0.14)")

        if accuracy is not None:
            kpi_card(c5, "📈", "Model Accuracy", f"{accuracy:.1%}", BLUE, "rgba(59,130,246,0.14)", "vs. real labels")
        else:
            try:
                import joblib
                model = joblib.load(MODEL_PATH)
                n_trees = getattr(model, "n_estimators", "N/A")
            except Exception:
                n_trees = "N/A"
            kpi_card(c5, "🌳", "Trees", str(n_trees), BLUE, "rgba(59,130,246,0.14)", "n_estimators")


# ---------------------------------------------------------------
# SECTION 8 - FOOTER
# ---------------------------------------------------------------

def _render_footer():
    with st.container(border=True, key="about_card_footer"):
        st.markdown(
            '<div class="about-footer">'
            '<div class="about-footer-title">PredictWise</div>'
            '<div class="about-footer-sub">Predictive Maintenance Decision Support System</div>'
            '<div class="about-footer-sub">Built with Python, Streamlit, and Machine Learning.</div>'
            '<div class="about-footer-tag">Designed to transform raw industrial sensor data '
            'into actionable maintenance insights.</div>'
            '</div>',
            unsafe_allow_html=True
        )