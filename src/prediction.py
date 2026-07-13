import joblib
import pandas as pd

MODEL_PATH = "models/random_forest.pkl"

model = joblib.load(MODEL_PATH)


def predict(X: pd.DataFrame):

    return model.predict(X)


def predict_probability(X: pd.DataFrame):

    return model.predict_proba(X)[:, 1]


def feature_importance(feature_names):

    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    )

    return importance.sort_values(
        "Importance",
        ascending=False,
    )