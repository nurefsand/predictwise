import pandas as pd


FEATURE_COLUMNS = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "temp_diff",
    "power",
    "wear_x_torque",
    "Type_H",
    "Type_L",
    "Type_M",
]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for model prediction.
    """

    df = pd.get_dummies(df, columns=["Type"])

    for column in ["Type_H", "Type_L", "Type_M"]:
        if column not in df.columns:
            df[column] = 0

    return df[FEATURE_COLUMNS]