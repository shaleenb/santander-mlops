import pytest
import pandas as pd
from backend.config import settings
from io import StringIO


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
def valid_file_object(sample_dataframe):
    string_buffer = StringIO()
    sample_dataframe.to_csv(string_buffer, index=False)
    string_buffer.seek(0)
    return string_buffer


@pytest.fixture
def invalid_file_object():
    string_buffer = StringIO()
    string_buffer.write("col1,col2\n1,2,3,4")
    string_buffer.seek(0)
    return string_buffer
