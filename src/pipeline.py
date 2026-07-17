"""
src/pipeline.py

Thin orchestration layer. It does NOT reimplement any ML logic -
it only chains the existing modules (feature_engineering,
preprocessing, prediction, business_rules) into one call, so that
every dashboard page (Dashboard, AI Prediction, ...) scores a
dataframe the same way instead of duplicating these five lines
in every page.

If you ever change how a machine gets scored, change it here once -
not in every page that needs predictions.
"""

import numpy as np
import pandas as pd

from src.feature_engineering import create_features
from src.preprocessing import prepare_data
from src.prediction import predict_probability
from src.business_rules import (
    calculate_status,
    calculate_health_score,
    maintenance_recommendation,
)


def score_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw machine-sensor dataframe (same columns as
    data/predictive_maintenance.csv: Type, Air temperature,
    Process temperature, Rotational speed, Torque, Tool wear)
    and returns it enriched with:

        Risk Score    - 0-100, higher = more likely to fail
        Confidence    - 0-100, how far the model's probability is
                        from the 50/50 decision boundary
        Status        - Healthy / Warning / Critical
        Health Score  - 0-100, higher = healthier
        Recommendation - short text, what to do about it

    ASSUMPTION TO VERIFY: this assumes predict_probability(X) returns
    a probability of failure in [0, 1] (as the original app.py implied
    via `risk_scores * 100`). If predict_probability actually returns
    something else (e.g. a multi-class array, or an already-scaled
    0-100 score), the Risk Score and Confidence lines below need a
    small adjustment - let me know what predict_probability's return
    shape looks like and I'll fix this in one place.
    """
    df = create_features(raw_df.copy())
    X = prepare_data(df)

    probability = predict_probability(X)

    df["Risk Score"] = probability * 100
    df["Confidence"] = np.maximum(probability, 1 - probability) * 100
    df["Status"] = df["Risk Score"].apply(calculate_status)
    df["Health Score"] = df["Risk Score"].apply(calculate_health_score)
    df["Recommendation"] = df["Risk Score"].apply(maintenance_recommendation)

    return df


def get_feature_matrix(raw_df: pd.DataFrame):
    """
    Runs the same feature engineering + preprocessing steps used by
    score_dataframe(), but returns the model-ready feature matrix X
    instead of enriched predictions.

    Added for dashboard/pages/explainability.py, which needs direct
    access to the exact matrix fed to the model (for SHAP and feature
    importance) without re-implementing the
    create_features -> prepare_data sequence a second time.
    """
    df = create_features(raw_df.copy())
    return prepare_data(df)