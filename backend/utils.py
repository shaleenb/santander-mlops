import pandas as pd
from backend.config import settings


def validate_data(df: pd.DataFrame, id_column: str) -> None:
    """
    Validates the uploaded data.

    Args:
        df (pd.DataFrame): The DataFrame containing the uploaded data.
        id_column (str): The name of the ID column in the uploaded file.

    Raises:
        ValueError: If the ID column is not found in the DataFrame.
        ValueError: If one or more feature columns are missing from the DataFrame.
        ValueError: If the type of the features in the DataFrame doesn't match the expected types.
    """
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in the uploaded file.")

    features = settings.MODEL_FEATURES

    if missing_features := set(features) - set(df.columns):
        missing_features_str = ", ".join(missing_features)
        raise ValueError(
            f"The following features are missing from the uploaded csv: {missing_features_str}"
        )

    non_numeric_features = [col for col in features if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_features:
        non_numeric_features_str = ", ".join(non_numeric_features)
        raise ValueError(f"The following features are non numeric {non_numeric_features_str}")
