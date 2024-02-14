from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd
from backend.api import app
from io import StringIO
import numpy as np

client = TestClient(app)


class MockModel:
    def predict(self, input_data):
        return np.zeros(len(input_data))

    def predict_proba(self, input_data):
        return np.array([[0, 1]] * len(input_data))


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_csv_read_error(empty_file):
    with open(empty_file, "rb") as file:
        response = client.post("/predict", files={"file": file})
    assert response.status_code == 400
    assert "An error occurred while reading the uploaded file" in response.json()["detail"]


def test_invalid_csv(bad_csv_file):
    with open(bad_csv_file, "rb") as file:
        response = client.post("/predict", files={"file": file})
    assert response.status_code == 400


@patch("backend.api.get_model", return_value=MockModel())
def test_predict_json(_, valid_csv_file):
    with open(valid_csv_file, "rb") as file:
        response = client.post("/predict?id_column=id&response_format=json", files={"file": file})
    assert response.status_code == 200
    assert response.json() == {str(i): {"Prediction": 0} for i in range(10)}


@patch("backend.api.get_model", return_value=MockModel())
def test_predict_json_confidence(_, valid_csv_file):
    with open(valid_csv_file, "rb") as file:
        response = client.post(
            "/predict?id_column=id&response_format=json&include_confidence=True",
            files={"file": file},
        )
    assert response.status_code == 200
    assert response.json() == {str(i): {"Prediction": 0, "Confidence": 1} for i in range(10)}


@patch("backend.api.get_model", return_value=MockModel())
def test_predict_csv(_, valid_csv_file):
    with open(valid_csv_file, "rb") as file:
        response = client.post("/predict?id_column=id&response_format=csv", files={"file": file})
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]

    df = pd.read_csv(StringIO(str(response.content, "utf-8")))
    assert df.shape == (10, 2)
    assert df.columns.tolist() == ["id", "Prediction"]
    assert df["Prediction"].tolist() == [0] * 10


@patch("backend.api.get_model", return_value=MockModel())
def test_predict_csv_confidence(_, valid_csv_file):
    with open(valid_csv_file, "rb") as file:
        response = client.post(
            "/predict?id_column=id&response_format=csv&include_confidence=True",
            files={"file": file},
        )
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]

    df = pd.read_csv(StringIO(str(response.content, "utf-8")))
    assert df.shape == (10, 3)
    assert df.columns.tolist() == ["id", "Prediction", "Confidence"]
    assert df["Prediction"].tolist() == [0] * 10
    assert df["Confidence"].tolist() == [1] * 10
