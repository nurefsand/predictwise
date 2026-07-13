import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for the predictive maintenance model.
    """

    df = df.copy()

    df["temp_diff"] = (
        df["Process temperature"]
        - df["Air temperature"]
    )

    df["power"] = (
        df["Torque"]
        * df["Rotational speed"]
    )

    df["wear_x_torque"] = (
        df["Tool wear"]
        * df["Torque"]
    )

    return df