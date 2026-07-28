"""
dashboard/views/about.py

Project introduction page. Mostly static/descriptive content, but
every number in Section 7 (Project Statistics) is read live from the
dataset, the feature pipeline, the trained model, or src/business_rules.py
- nothing there is a hardcoded/invented figure.

Card sections use dashboard.layout.section() and the shared
card_grid()/grid_card()/workflow_strip() helpers instead of a
page-local flexbox grid, so tech/feature cards get real equal-height
rows (4 desktop / 2 tablet / 1 mobile) from theme.py's .card-grid,
not a one-off About-page-only layout.
"""

from pathlib import Path

import streamlit as st
import pandas as pd

from dashboard.theme import BLUE, TEXT_PRIMARY, TEXT_SECONDARY, BORDER, kpi_card
from dashboard.layout import section, spacer, card_grid, grid_card, workflow_strip
from dashboard.data import load_scored_fleet
from src.pipeline import get_feature_matrix
from src.business_rules import calculate_status

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "predictive_maintenance.csv"

CANDIDATE_LABEL_COLUMNS = [
    "Machine failure", "machine failure", "Machine Failure", "Target", "target",
]

# Only the bits that are genuinely unique to this page (hero text,
# architecture tree, footer) - everything reusable (card grids,
# section cards, workflow strip) now lives in theme.py.
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
    spacer()
    _render_overview()
    spacer()
    _render_workflow()
    spacer()
    _render_architecture()
    spacer()
    _render_tech_stack()
    spacer()
    _render_features()
    spacer()
    _render_statistics()
    spacer()
    _render_footer()


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
    with section("Project Overview"):
        st.markdown(
            '<div class="summary-line">'
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
    with section("How PredictWise Works"):
        workflow_strip(steps)


# ---------------------------------------------------------------
# SECTION 4 - PROJECT ARCHITECTURE
# ---------------------------------------------------------------

def _render_architecture():
    groups = [
        ("dashboard/", ["Streamlit application (routing, theme, cached data access)"]),
        ("dashboard/views/", [
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

    with section("Project Architecture"):
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
    with section("Technology Stack"):
        cards = [grid_card(icon, name, desc) for icon, name, desc in stack]
        card_grid(cards, columns=4)


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
    with section("Application Features"):
        cards = [grid_card(icon, title, desc) for icon, title, desc in features]
        card_grid(cards, columns=4)


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

    with section("Project Statistics"):
        from dashboard.theme import HEALTHY, WARNING, CRITICAL

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
    with section("PredictWise"):
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