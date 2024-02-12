import pytest
from backend.utils import validate_data


def test_validate_data_with_valid_data(sample_dataframe):
    id_column = "id"
    assert validate_data(sample_dataframe, id_column) is None


def test_validate_data_with_missing_id_column(sample_dataframe):
    id_column = "nonexistent_column"
    with pytest.raises(ValueError) as exc_info:
        validate_data(sample_dataframe, id_column)
    assert str(exc_info.value) == "ID column 'nonexistent_column' not found in the uploaded file."


def test_validate_data_with_missing_feature_columns(sample_dataframe):
    sample_dataframe.drop(columns=["var_1", "var_2"], inplace=True)
    id_column = "id"
    with pytest.raises(ValueError) as exc_info:
        validate_data(sample_dataframe, id_column)
    assert (
        str(exc_info.value)
        == "The following features are missing from the uploaded csv: var_1, var_2"
    )


def test_validate_data_with_invalid_feature_types(sample_dataframe):
    sample_dataframe["var_1"] = "a"
    id_column = "id"
    with pytest.raises(ValueError) as exc_info:
        validate_data(sample_dataframe, id_column)
    assert str(exc_info.value) == "The following features are non numeric var_1"
