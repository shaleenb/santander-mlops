import pandas as pd
import pytest

from backend.config import settings


@pytest.fixture
def sample_dataframe():
    dataset_length = 10
    features = settings.MODEL_FEATURES
    data = {
        "id": range(dataset_length),
        **{feature: range(dataset_length) for feature in features},
    }
    return pd.DataFrame(data)


@pytest.fixture
def valid_csv_file(tmp_path, sample_dataframe):
    file_path = tmp_path / "sample.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def bad_csv_file(tmp_path):
    file_path = tmp_path / "invalid.csv"
    with open(file_path, "w") as file:
        file.write("column1,column2\nvalue1,value2")
    return file_path


@pytest.fixture
def empty_file(tmp_path):
    file_path = tmp_path / "invalid.csv"
    with open(file_path, "w") as file:
        file.write("")
    return file_path
