"""
dashboard/pages/explainability.py

Explainable AI page: global feature importance, SHAP summary plot,
per-feature exploration, and per-machine local explanations.

Design note on src/explainability.py reuse: this file does not know
the exact function names inside src/explainability.py (it wasn't
shared), so every SHAP/model call below first tries to reuse a
matching function from src.explainability if one exists, and only
falls back to computing it directly with the `shap` library if it
doesn't. Search for "REUSE ATTEMPT" comments to see exactly where -
if src/explainability.py already has get_feature_importance(),
get_shap_values(), explain_instance(), etc., point them out and this
file will use them directly instead of the fallback path.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dashboard.theme import (
    HEALTHY, WARNING, CRITICAL, BLUE,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER, CARD,
    page_title, kpi_card, base_plotly_layout, badge_html,
)
from dashboard.data import load_scored_fleet
from src.pipeline import get_feature_matrix

MODEL_PATH = "models/random_forest.pkl"

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# REUSE ATTEMPT: use src/explainability.py's own functions if they exist
try:
    import src.explainability as expl_module
except ImportError:
    expl_module = None


# =========================================================
# MODEL / DATA ACCESS
# =========================================================

@st.cache_resource(show_spinner="Loading trained model...")
def _load_model():
    """Loads the trained Random Forest model. Tries joblib first
    (the common convention for sklearn .pkl files), falls back to
    pickle. If src/explainability.py already exposes a load_model()
    or similar, that's the better source of truth - tell me its name
    and I'll swap this to call it instead."""
    import joblib
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        import pickle
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)


@st.cache_data(show_spinner="Preparing feature matrix...")
def _get_sample_matrix(sample_size: int = 500, random_state: int = 42):
    """Samples the fleet for SHAP (full 10k rows would be slow to
    explain on every rerun). Returns (raw_sample_df, X_sample)."""
    df = load_scored_fleet()
    sample_df = df.sample(min(sample_size, len(df)), random_state=random_state)
    X = get_feature_matrix(sample_df)
    return sample_df.reset_index(drop=True), X


def _feature_names(X) -> list:
    if isinstance(X, pd.DataFrame):
        return X.columns.tolist()
    n = X.shape[1] if hasattr(X, "shape") else len(X[0])
    return [f"Feature_{i}" for i in range(n)]


def _positive_class_shap(shap_values):
    """Normalizes different SHAP return shapes to 'contribution
    toward failure/risk' for a binary classifier."""
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    if hasattr(shap_values, "values"):
        vals = shap_values.values
        if vals.ndim == 3:
            return vals[:, :, 1]
        return vals
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


@st.cache_data(show_spinner="Computing SHAP values (one-time, cached)...")
def _compute_shap(sample_size: int = 500):
    """Returns (X_sample, shap_values_for_failure_class) or (None, None)
    if SHAP/model isn't available."""
    if not SHAP_AVAILABLE:
        return None, None
    try:
        model = _load_model()
        _, X = _get_sample_matrix(sample_size)

        if expl_module is not None and hasattr(expl_module, "get_shap_values"):
            # REUSE ATTEMPT
            raw_shap = expl_module.get_shap_values(model, X)
        else:
            explainer = shap.TreeExplainer(model)
            raw_shap = explainer.shap_values(X)

        return X, _positive_class_shap(raw_shap)
    except Exception as e:
        st.session_state["_shap_error"] = str(e)
        return None, None


def _get_feature_importance(model, feature_names: list) -> pd.Series:
    if expl_module is not None and hasattr(expl_module, "get_feature_importance"):
        # REUSE ATTEMPT
        try:
            return expl_module.get_feature_importance(model, feature_names)
        except Exception:
            pass
    importances = model.feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def _apply_dark_matplotlib():
    if MATPLOTLIB_AVAILABLE:
        plt.rcParams.update({
            "figure.facecolor": CARD,
            "axes.facecolor": CARD,
            "savefig.facecolor": CARD,
            "axes.edgecolor": BORDER,
            "text.color": TEXT_PRIMARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
        })


# =========================================================
# PAGE
# =========================================================

def show():
    page_title("🧠", "Explainable AI", "Understand why the model makes its decisions")

    if not SHAP_AVAILABLE:
        st.error(
            "The `shap` package isn't importable in this environment "
            "(`pip install shap`). Sections that depend on it will be skipped; "
            "feature importance from the model itself still works."
        )

    _render_summary_card()
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    model = _load_model()
    _, X_sample = _get_sample_matrix()
    feature_names = _feature_names(X_sample)
    importance = _get_feature_importance(model, feature_names)

    _render_global_importance(importance)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_shap_summary()
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_feature_explorer()
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_local_explanation(model, feature_names)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_decision_process()
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_key_insights(importance)


# ---------------------------------------------------------------
# SECTION 1 - MODEL EXPLANATION SUMMARY
# ---------------------------------------------------------------

def _render_summary_card():
    lines = (
        "• <b>Model:</b> Random Forest classifier, trained on historical machine sensor readings<br>"
        "• <b>Explainability method:</b> SHAP (SHapley Additive exPlanations)<br>"
        "• <b>Global explanations:</b> which sensors matter most across the whole fleet<br>"
        "• <b>Local explanations:</b> why one specific machine got its risk score<br>"
        "• <b>Feature importance:</b> a ranked score per sensor, from the trained model directly"
    )
    st.markdown(
        '<div class="summary-panel"><div class="summary-title">Model Explanation Summary</div>'
        f'<div class="summary-line">{lines}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------
# SECTION 2 - GLOBAL FEATURE IMPORTANCE
# ---------------------------------------------------------------

def _render_global_importance(importance: pd.Series):
    st.markdown(
        '<div class="chart-card"><div class="chart-title">Global Feature Importance</div>'
        '<div class="chart-note">Ranked by the trained Random Forest\'s own importance scores</div>',
        unsafe_allow_html=True
    )

    imp_df = importance.reset_index()
    imp_df.columns = ["Feature", "Importance"]
    imp_df = imp_df.sort_values("Importance", ascending=True)

    fig = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        text="Importance", color="Importance",
        color_continuous_scale=[[0, BLUE], [1, WARNING]],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                       textfont=dict(color=TEXT_PRIMARY, size=11))
    fig.update_layout(
        **base_plotly_layout(height=max(260, 34 * len(imp_df))),
        coloraxis_showscale=False,
        xaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER),
        yaxis=dict(color=TEXT_SECONDARY, showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 3 - SHAP SUMMARY PLOT
# ---------------------------------------------------------------

def _render_shap_summary():
    st.markdown(
        '<div class="chart-card"><div class="chart-title">SHAP Summary Plot</div>',
        unsafe_allow_html=True
    )

    if not (SHAP_AVAILABLE and MATPLOTLIB_AVAILABLE):
        st.warning("SHAP or matplotlib isn't available, so the summary plot can't be rendered.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    X_sample, shap_vals = _compute_shap()

    if shap_vals is None:
        st.warning(
            "Couldn't compute SHAP values: "
            f"{st.session_state.get('_shap_error', 'unknown error')}. "
            "This usually means the model path, the feature matrix shape, or the SHAP "
            "API version doesn't match what this page assumes - let me know the error "
            "and I'll adjust it."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    _apply_dark_matplotlib()
    fig = plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_vals, X_sample, show=False, plot_size=None)
    plt.gcf().set_facecolor(CARD)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close(fig)

    st.markdown(
        '<div class="chart-note" style="margin-top:8px;">'
        '🔴 Red = higher feature values &nbsp;·&nbsp; 🔵 Blue = lower feature values &nbsp;·&nbsp; '
        'Features at the top influence predictions the most.</div>',
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 4 - FEATURE IMPACT EXPLORER
# ---------------------------------------------------------------

def _render_feature_explorer():
    st.markdown(
        '<div class="chart-card"><div class="chart-title">Feature Impact Explorer</div>',
        unsafe_allow_html=True
    )

    df = load_scored_fleet()
    numeric_cols = [
        "Air temperature", "Process temperature", "Rotational speed",
        "Torque", "Tool wear", "Risk Score", "Health Score",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    selected = st.selectbox("Select a feature", numeric_cols, label_visibility="collapsed")

    series = df[selected]
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "μ", "Average", f"{series.mean():.2f}", BLUE, "rgba(59,130,246,0.14)")
    kpi_card(c2, "▼", "Min", f"{series.min():.2f}", HEALTHY, "rgba(34,197,94,0.14)")
    kpi_card(c3, "▲", "Max", f"{series.max():.2f}", CRITICAL, "rgba(239,68,68,0.14)")
    kpi_card(c4, "σ", "Std Dev", f"{series.std():.2f}", WARNING, "rgba(245,158,11,0.14)")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    fig = px.histogram(df, x=selected, nbins=30, color_discrete_sequence=[BLUE])
    fig.update_layout(
        **base_plotly_layout(height=260),
        xaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER),
        yaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER, title="Machines"),
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 5 - LOCAL PREDICTION EXPLANATION
# ---------------------------------------------------------------

def _render_local_explanation(model, feature_names: list):
    st.markdown(
        '<div class="chart-card"><div class="chart-title">Local Prediction Explanation</div>'
        '<div class="chart-note">Why did this specific machine get its risk score?</div>',
        unsafe_allow_html=True
    )

    df = load_scored_fleet()
    row_number = st.number_input(
        "Row number", min_value=0, max_value=len(df) - 1, value=0, step=1,
        help="Row index in the fleet dataset (0-based).",
    )

    row = df.iloc[[row_number]]
    fg, _ = (CRITICAL, None) if row["Status"].iloc[0] == "Critical" else \
            (WARNING, None) if row["Status"].iloc[0] == "Warning" else (HEALTHY, None)

    c1, c2, c3 = st.columns(3)
    kpi_card(c1, "🎯", "Prediction", row["Status"].iloc[0], fg, "rgba(148,163,184,0.14)")
    kpi_card(c2, "⚠️", "Risk Score", f'{row["Risk Score"].iloc[0]:.1f}', fg, "rgba(148,163,184,0.14)")
    kpi_card(c3, "💙", "Health Score", f'{row["Health Score"].iloc[0]:.1f}', fg, "rgba(148,163,184,0.14)")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    if not SHAP_AVAILABLE:
        st.warning("SHAP isn't available, so per-machine contributions can't be computed.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    try:
        X_row = get_feature_matrix(row)
        explainer = shap.TreeExplainer(model)

        waterfall_rendered = False
        try:
            explanation = explainer(X_row)
            values = explanation.values
            if values.ndim == 3:
                values = values[:, :, 1]
            single = shap.Explanation(
                values=values[0],
                base_values=explanation.base_values[0]
                if np.ndim(explanation.base_values) > 0 else explanation.base_values,
                data=X_row.iloc[0] if isinstance(X_row, pd.DataFrame) else X_row[0],
                feature_names=feature_names,
            )
            _apply_dark_matplotlib()
            fig = plt.figure(figsize=(9, 4.5))
            shap.plots.waterfall(single, show=False, max_display=10)
            plt.gcf().set_facecolor(CARD)
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.close(fig)
            waterfall_rendered = True
        except Exception:
            waterfall_rendered = False

        if not waterfall_rendered:
            # Fallback: ranked horizontal contribution chart in Plotly,
            # built from the raw shap_values() output instead of the
            # newer Explanation-based waterfall API.
            raw_shap = explainer.shap_values(X_row)
            contrib = _positive_class_shap(raw_shap)[0]
            contrib_df = pd.DataFrame({"Feature": feature_names, "Contribution": contrib})
            contrib_df = contrib_df.reindex(
                contrib_df["Contribution"].abs().sort_values(ascending=True).index
            )

            fig = px.bar(
                contrib_df, x="Contribution", y="Feature", orientation="h",
                color="Contribution", color_continuous_scale=[[0, HEALTHY], [0.5, "#171b22"], [1, CRITICAL]],
            )
            fig.update_layout(
                **base_plotly_layout(height=max(260, 30 * len(contrib_df))),
                coloraxis_showscale=False,
                xaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER, title="Contribution to risk"),
                yaxis=dict(color=TEXT_SECONDARY, showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)

            top_positive = contrib_df.sort_values("Contribution", ascending=False).head(3)
            top_negative = contrib_df.sort_values("Contribution", ascending=True).head(3)

            colc1, colc2 = st.columns(2)
            with colc1:
                st.markdown("**Top features increasing risk**")
                for _, r in top_positive.iterrows():
                    st.markdown(f"- {r['Feature']}: +{r['Contribution']:.3f}")
            with colc2:
                st.markdown("**Top features reducing risk**")
                for _, r in top_negative.iterrows():
                    st.markdown(f"- {r['Feature']}: {r['Contribution']:.3f}")

    except Exception as e:
        st.warning(
            f"Couldn't generate a local explanation for this machine: {e}. "
            "This is likely a mismatch between the assumed feature matrix shape "
            "and what the model actually expects - share src/preprocessing.py "
            "and I'll fix it precisely."
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 6 - MODEL DECISION PROCESS
# ---------------------------------------------------------------

def _render_decision_process():
    steps = [
        ("📡", "Sensor Data", "Air temp, process temp, rotational speed, torque, tool wear"),
        ("🛠️", "Feature Engineering", "Derived signals computed from raw sensors"),
        ("🌲", "Random Forest", "Trained classifier scores the machine"),
        ("📈", "Probability", "Model's raw failure probability"),
        ("⚠️", "Risk Score", "Probability scaled to 0-100"),
        ("💙", "Health Score", "Inverse view of risk, 0-100"),
        ("✅", "Recommendation", "Action for the maintenance engineer"),
    ]

    rows_html = ""
    for i, (icon, title, desc) in enumerate(steps):
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:14px;padding:10px 4px;">'
            f'<div style="width:34px;height:34px;border-radius:8px;background:rgba(59,130,246,0.14);'
            f'display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;">{icon}</div>'
            f'<div><div style="color:{TEXT_PRIMARY};font-weight:600;font-size:13.5px;">{title}</div>'
            f'<div style="color:{TEXT_SECONDARY};font-size:11.5px;">{desc}</div></div></div>'
        )
        if i < len(steps) - 1:
            rows_html += (
                f'<div style="text-align:center;color:{TEXT_SECONDARY};font-size:14px;">↓</div>'
            )

    st.markdown(
        '<div class="chart-card"><div class="chart-title">Model Decision Process</div>'
        f'{rows_html}</div>',
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------
# SECTION 7 - KEY INSIGHTS (generated from real importance/correlation)
# ---------------------------------------------------------------

def _render_key_insights(importance: pd.Series):
    df = load_scored_fleet()
    ranked = importance.sort_values(ascending=False)
    top_feature = ranked.index[0]
    least_feature = ranked.index[-1]

    insights = [f"<b>{top_feature}</b> has the strongest influence on predicted failures "
                f"(importance: {ranked.iloc[0]:.3f})."]

    for feature in ranked.index[1:4]:
        if feature in df.columns:
            corr = df[feature].corr(df["Risk Score"])
            if pd.notna(corr) and abs(corr) > 0.05:
                direction = "increases" if corr > 0 else "decreases"
                insights.append(
                    f"Higher <b>{feature}</b> tends to {direction} predicted risk "
                    f"(correlation with Risk Score: {corr:+.2f})."
                )

    insights.append(
        f"<b>{least_feature}</b> has relatively small influence on the model's decisions "
        f"(importance: {ranked.iloc[-1]:.3f})."
    )

    cols = st.columns(2)
    for i, insight in enumerate(insights[:6]):
        with cols[i % 2]:
            st.markdown(
                '<div class="chart-card" style="margin-bottom:14px;">'
                f'<div class="summary-line">💡 {insight}</div></div>',
                unsafe_allow_html=True
            )