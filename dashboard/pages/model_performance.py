"""
dashboard/pages/model_performance.py

Evaluates the already-trained Random Forest model. Never retrains it.

READ THIS BEFORE TRUSTING THE NUMBERS ON THIS PAGE:

Real Accuracy/Precision/Recall/F1/ROC-AUC/Confusion-Matrix require a
TRUE label to compare predictions against. This app's predicted
"Status" (Healthy/Warning/Critical) is a 3-way bucket built by
thresholding one failure probability - it is not itself ground truth,
and as far as I can see, this project's data only carries a BINARY
true label (the AI4I 2020 dataset's "Machine failure" 0/1 column, if
present in data/predictive_maintenance.csv). There is no genuine
"this machine was truly in a Warning state" label anywhere - inventing
one to force a 3x3 confusion matrix would mean fabricating labels,
which directly contradicts "no fake metrics / no mock data".

So, if a binary ground-truth column is found:
  - KPIs, ROC curve, and classification report are computed as a real
    BINARY evaluation (Failure vs No Failure).
  - The "confusion matrix" is Actual (No Failure / Failure) x
    Predicted Status (Healthy / Warning / Critical) - a real, honest
    2x3 grid - instead of a fabricated symmetric 3x3.

If no such column is found, this page says so explicitly instead of
making up numbers, and still shows the sections that don't need
ground truth (confidence distribution, feature importance, model
info).

If your training pipeline actually has real Warning/Critical severity
labels (e.g. saved from src/train_model.py's original train/test
split), tell me where and I'll wire this page to that instead.
"""

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
)

from dashboard.theme import (
    HEALTHY, WARNING, CRITICAL, BLUE,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER, CARD,
    page_title, kpi_card, base_plotly_layout,
)
from dashboard.data import load_scored_fleet
from src.pipeline import get_feature_matrix

# Paths resolved relative to the project root, not the working
# directory the app happens to be launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "predictive_maintenance.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"

CANDIDATE_LABEL_COLUMNS = [
    "Machine failure", "machine failure", "Machine Failure",
    "Target", "target", "Failure", "failure",
]


# =========================================================
# DATA / MODEL ACCESS (cached, self-contained, pathlib-based)
# =========================================================

@st.cache_resource(show_spinner="Loading trained model...")
def _load_trained_model():
    """Loads the already-trained Random Forest. Never fits a new one."""
    import joblib
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        import pickle
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)


@st.cache_data(show_spinner=False)
def _load_ground_truth() -> pd.Series | None:
    """Reads the raw CSV directly to look for a real binary failure
    label, independent of the prediction pipeline. Returns a 0/1
    Series aligned to row order, or None if no candidate column
    exists in the file."""
    if not DATA_PATH.exists():
        return None
    raw = pd.read_csv(DATA_PATH)
    for col in CANDIDATE_LABEL_COLUMNS:
        if col in raw.columns:
            return raw[col].astype(int).reset_index(drop=True)
    return None


@st.cache_data(show_spinner="Evaluating model...")
def _build_evaluation_bundle():
    """Computes every real metric used on this page, once, from the
    already-scored fleet and the real ground-truth column (if any)."""
    df = load_scored_fleet().reset_index(drop=True)
    y_true = _load_ground_truth()

    if y_true is None or len(y_true) != len(df):
        return {"available": False, "df": df}

    y_prob = (df["Risk Score"] / 100).to_numpy()
    y_pred = (df["Status"] != "Healthy").astype(int).to_numpy()
    y_true_arr = y_true.to_numpy()

    report = classification_report(
        y_true_arr, y_pred, labels=[0, 1],
        target_names=["No Failure", "Failure"], output_dict=True, zero_division=0,
    )
    fpr, tpr, _ = roc_curve(y_true_arr, y_prob)
    auc_value = roc_auc_score(y_true_arr, y_prob)

    true_label = pd.Series(y_true_arr).map({0: "No Failure", 1: "Failure"})
    cross = pd.crosstab(true_label, df["Status"])
    cross = cross.reindex(index=["No Failure", "Failure"],
                           columns=["Healthy", "Warning", "Critical"], fill_value=0)

    return {
        "available": True,
        "df": df,
        "y_true": y_true_arr,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": accuracy_score(y_true_arr, y_pred),
        "precision": precision_score(y_true_arr, y_pred, zero_division=0),
        "recall": recall_score(y_true_arr, y_pred, zero_division=0),
        "f1": f1_score(y_true_arr, y_pred, zero_division=0),
        "roc_auc": auc_value,
        "report": report,
        "fpr": fpr, "tpr": tpr,
        "cross": cross,
    }


def _get_feature_importance() -> pd.Series:
    model = _load_trained_model()
    _, X = load_scored_fleet(), get_feature_matrix(load_scored_fleet().head(5))
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else \
        [f"Feature_{i}" for i in range(len(model.feature_importances_))]
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)


# =========================================================
# PAGE
# =========================================================

def show():
    page_title("📈", "Model Performance", "Evaluate the predictive quality of the Random Forest model")

    bundle = _build_evaluation_bundle()

    if not bundle["available"]:
        _render_no_ground_truth_notice()
        _render_confidence_distribution(bundle["df"])
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        _render_feature_importance()
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        _render_validation_summary(bundle["df"], ground_truth_found=False)
        return

    _render_kpis(bundle)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_confusion_matrix(bundle)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_classification_report(bundle)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_roc_curve(bundle)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_precision_recall_bars(bundle)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_confidence_distribution(bundle["df"])
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_feature_importance()
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_validation_summary(bundle["df"], ground_truth_found=True)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _render_insights(bundle)


def _render_no_ground_truth_notice():
    st.warning(
        "No ground-truth failure column was found in `data/predictive_maintenance.csv` "
        f"(looked for: {', '.join(CANDIDATE_LABEL_COLUMNS)}). "
        "Accuracy, Precision, Recall, F1, ROC-AUC, the confusion matrix and the "
        "classification report all require a true label to compare against, so "
        "they're skipped here rather than showing fabricated numbers. "
        "Confidence distribution, feature importance and the model summary below "
        "don't need ground truth, so they're still shown. "
        "If the true label lives somewhere else (a different column name, or a "
        "held-out test set saved by src/train_model.py), point me to it and I'll "
        "wire this page up to it."
    )


# ---------------------------------------------------------------
# SECTION 1 - PERFORMANCE SUMMARY (KPIs)
# ---------------------------------------------------------------

def _render_kpis(bundle: dict):
    row1 = st.columns(3)
    kpi_card(row1[0], "🎯", "Accuracy", f"{bundle['accuracy']:.1%}", BLUE, "rgba(59,130,246,0.14)", "Binary: Failure vs No Failure")
    kpi_card(row1[1], "🎯", "Precision", f"{bundle['precision']:.1%}", HEALTHY, "rgba(34,197,94,0.14)", "Of predicted failures, % correct")
    kpi_card(row1[2], "🎯", "Recall", f"{bundle['recall']:.1%}", WARNING, "rgba(245,158,11,0.14)", "Of real failures, % caught")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    row2 = st.columns(3)
    kpi_card(row2[0], "🎯", "F1 Score", f"{bundle['f1']:.1%}", BLUE, "rgba(59,130,246,0.14)", "Precision/Recall balance")
    kpi_card(row2[1], "📈", "ROC-AUC", f"{bundle['roc_auc']:.3f}", CRITICAL, "rgba(239,68,68,0.14)", "1.0 = perfect ranking")
    kpi_card(row2[2], "🏭", "Predictions", f"{len(bundle['df']):,}", TEXT_SECONDARY, "rgba(148,163,184,0.14)", "Machines scored")


# ---------------------------------------------------------------
# SECTION 2 - CONFUSION MATRIX
# ---------------------------------------------------------------

def _render_confusion_matrix(bundle: dict):
    st.markdown(
        '<div class="chart-card"><div class="chart-title">Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    cross = bundle["cross"]

    fig = px.imshow(
        cross.values, text_auto=True,
        x=cross.columns.tolist(), y=cross.index.tolist(),
        color_continuous_scale=[[0, "#171b22"], [1, BLUE]],
        aspect="auto",
    )
    fig.update_traces(textfont=dict(size=14, color=TEXT_PRIMARY))
    fig.update_layout(
        **base_plotly_layout(height=260),
        xaxis=dict(title="Predicted Status", color=TEXT_SECONDARY, side="bottom"),
        yaxis=dict(title="Actual Outcome", color=TEXT_SECONDARY),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="model_perf_confusion_matrix")

    st.markdown(
        '<div class="chart-note">Rows are the real outcome (from the dataset\'s true '
        "failure label). Columns are what the model predicted. A well-performing model "
        "should show actual Failures concentrated under Warning/Critical, and actual "
        "No-Failures concentrated under Healthy.</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 3 - CLASSIFICATION REPORT
# ---------------------------------------------------------------

def _render_classification_report(bundle: dict):
    st.markdown(
        '<div class="chart-card"><div class="chart-title">Classification Report</div>'
        '<div class="chart-note">Binary evaluation: Warning + Critical are both treated '
        'as "predicted Failure" here, compared against the real failure label.</div>',
        unsafe_allow_html=True
    )

    report = bundle["report"]
    rows = []
    for label in ["No Failure", "Failure", "macro avg", "weighted avg"]:
        r = report[label]
        rows.append({
            "Class": label,
            "Precision": round(r["precision"], 3),
            "Recall": round(r["recall"], 3),
            "F1-score": round(r["f1-score"], 3),
            "Support": int(r["support"]),
        })
    report_df = pd.DataFrame(rows)

    st.dataframe(report_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 4 - ROC CURVE
# ---------------------------------------------------------------

def _render_roc_curve(bundle: dict):
    st.markdown(
        '<div class="chart-card"><div class="chart-title">ROC Curve</div>'
        '<div class="chart-note">Ground truth here is binary (Failure vs No Failure), '
        "so this is a single ROC curve rather than multiclass one-vs-rest.</div>",
        unsafe_allow_html=True
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bundle["fpr"], y=bundle["tpr"], mode="lines",
        name=f"Failure (AUC = {bundle['roc_auc']:.3f})",
        line=dict(color=BLUE, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random guess", line=dict(color=TEXT_SECONDARY, width=1, dash="dash"),
    ))
    fig.update_layout(
        **base_plotly_layout(height=340),
        xaxis=dict(title="False Positive Rate", color=TEXT_SECONDARY, gridcolor=BORDER, range=[0, 1]),
        yaxis=dict(title="True Positive Rate", color=TEXT_SECONDARY, gridcolor=BORDER, range=[0, 1]),
        legend=dict(font=dict(color=TEXT_PRIMARY, size=11), x=0.55, y=0.08),
    )
    st.plotly_chart(fig, use_container_width=True, key="model_perf_roc_curve")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 5 - PRECISION VS RECALL BY CLASS
# ---------------------------------------------------------------

def _render_precision_recall_bars(bundle: dict):
    report = bundle["report"]
    classes = ["No Failure", "Failure"]
    precision_vals = [report[c]["precision"] for c in classes]
    recall_vals = [report[c]["recall"] for c in classes]

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="chart-card"><div class="chart-title">Precision by Class</div>',
            unsafe_allow_html=True
        )
        fig = px.bar(x=classes, y=precision_vals, text=[f"{v:.1%}" for v in precision_vals],
                     color=classes, color_discrete_sequence=[HEALTHY, CRITICAL])
        fig.update_traces(textposition="outside", textfont=dict(color=TEXT_PRIMARY))
        fig.update_layout(
            **base_plotly_layout(height=260), showlegend=False,
            xaxis=dict(color=TEXT_SECONDARY, title=""),
            yaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER, range=[0, 1.1]),
        )
        st.plotly_chart(fig, use_container_width=True, key="model_perf_precision_by_class")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="chart-card"><div class="chart-title">Recall by Class</div>',
            unsafe_allow_html=True
        )
        fig = px.bar(x=classes, y=recall_vals, text=[f"{v:.1%}" for v in recall_vals],
                     color=classes, color_discrete_sequence=[HEALTHY, CRITICAL])
        fig.update_traces(textposition="outside", textfont=dict(color=TEXT_PRIMARY))
        fig.update_layout(
            **base_plotly_layout(height=260), showlegend=False,
            xaxis=dict(color=TEXT_SECONDARY, title=""),
            yaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER, range=[0, 1.1]),
        )
        st.plotly_chart(fig, use_container_width=True, key="model_perf_recall_by_class")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 6 - PREDICTION CONFIDENCE DISTRIBUTION
# ---------------------------------------------------------------

def _render_confidence_distribution(df: pd.DataFrame):
    st.markdown(
        '<div class="chart-card"><div class="chart-title">Prediction Confidence Distribution</div>',
        unsafe_allow_html=True
    )

    fig = px.histogram(df, x="Confidence", nbins=30, color_discrete_sequence=[BLUE])
    fig.update_layout(
        **base_plotly_layout(height=260),
        xaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER, title="Confidence (%)"),
        yaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER, title="Machines"),
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True, key="model_perf_confidence_dist")

    st.markdown(
        '<div class="chart-note">Confidence reflects how far the model\'s probability is '
        "from the 50/50 decision boundary. Higher confidence generally means a more "
        "reliable prediction; predictions clustered near 50% are the ones worth a second "
        "look from an engineer.</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 7 - FEATURE IMPORTANCE
# ---------------------------------------------------------------

def _render_feature_importance():
    st.markdown(
        '<div class="chart-card"><div class="chart-title">Feature Importance</div>'
        '<div class="chart-note">From the trained Random Forest\'s own importance scores</div>',
        unsafe_allow_html=True
    )

    importance = _get_feature_importance()
    pct = (importance / importance.sum() * 100).reset_index()
    pct.columns = ["Feature", "Importance %"]
    pct = pct.sort_values("Importance %", ascending=True)

    fig = px.bar(
        pct, x="Importance %", y="Feature", orientation="h", text="Importance %",
        color="Importance %", color_continuous_scale=[[0, BLUE], [1, WARNING]],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                       textfont=dict(color=TEXT_PRIMARY, size=11))
    fig.update_layout(
        **base_plotly_layout(height=max(260, 34 * len(pct))),
        coloraxis_showscale=False,
        xaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER),
        yaxis=dict(color=TEXT_SECONDARY, showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True, key="model_perf_feature_importance")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# SECTION 8 - MODEL VALIDATION SUMMARY
# ---------------------------------------------------------------

def _render_validation_summary(df: pd.DataFrame, ground_truth_found: bool):
    gt_line = (
        "Evaluated against the dataset's real binary failure label."
        if ground_truth_found else
        "No ground-truth label found - evaluation metrics skipped, see notice above."
    )
    lines = (
        "• <b>Algorithm:</b> Random Forest<br>"
        f"• <b>Dataset:</b> {len(df):,} machines<br>"
        "• <b>Predicted classes:</b> Healthy / Warning / Critical "
        "(thresholded from one failure probability)<br>"
        "• <b>Features used:</b> Type, Air Temperature, Process Temperature, "
        "Rotational Speed, Torque, Tool Wear<br>"
        "• <b>Output:</b> Risk Score, Health Score, Recommendation<br>"
        f"• <b>Validation:</b> {gt_line}"
    )
    st.markdown(
        '<div class="summary-panel"><div class="summary-title">Model Validation Summary</div>'
        f'<div class="summary-line">{lines}</div></div>',
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------
# SECTION 9 - PERFORMANCE INSIGHTS (generated from real metrics)
# ---------------------------------------------------------------

def _render_insights(bundle: dict):
    insights = []

    acc = bundle["accuracy"]
    if acc >= 0.95:
        insights.append(f"Excellent overall accuracy ({acc:.1%}) on the binary failure task.")
    elif acc >= 0.85:
        insights.append(f"Good overall accuracy ({acc:.1%}), with room to improve on harder cases.")
    else:
        insights.append(f"Accuracy is moderate ({acc:.1%}) - worth reviewing which cases are missed.")

    cross = bundle["cross"]
    healthy_precision = cross.loc["No Failure", "Healthy"] / max(cross["Healthy"].sum(), 1)
    insights.append(
        f"Of machines predicted Healthy, {healthy_precision:.1%} were truly non-failing "
        "in the real data."
    )

    critical_recall = cross.loc["Failure", "Critical"] / max(cross.loc["Failure"].sum(), 1)
    insights.append(
        f"{critical_recall:.1%} of real failures were flagged as Critical by the model."
    )

    off_diag = cross.copy()
    off_diag.loc["No Failure", "Healthy"] = 0
    off_diag.loc["Failure", "Critical"] = 0
    if off_diag.to_numpy().sum() > 0:
        r_idx, c_idx = np.unravel_index(off_diag.to_numpy().argmax(), off_diag.shape)
        insights.append(
            f"Most classification errors occur between actual '{off_diag.index[r_idx]}' "
            f"machines predicted as '{off_diag.columns[c_idx]}'."
        )

    cols = st.columns(2)
    for i, insight in enumerate(insights[:6]):
        with cols[i % 2]:
            st.markdown(
                '<div class="chart-card" style="margin-bottom:14px;">'
                f'<div class="summary-line">💡 {insight}</div></div>',
                unsafe_allow_html=True
            )