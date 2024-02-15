import os
from uuid import uuid4

import joblib
import pandas as pd
from backend.config import settings
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from backend.utils import validate_data

app = FastAPI()

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(settings.MODEL_PATH)
    return _model


@app.post("/predict")
def predict(
    file: UploadFile = File(..., description="The CSV file containing the data for prediction."),
    id_column: str | None = Query(
        None,
        alias="id_column",
        description="The name of the ID column in the CSV file.",
    ),
    response_format: str = Query(
        "json",
        alias="response_format",
        description="The format in which the predictions should be returned.",
    ),
    include_confidence: bool = Query(
        False,
        alias="include_confidence",
        description="Whether to include confidence scores in the predictions.",
    ),
):
    """
    Makes predictions on the uploaded file.

    **Parameters**:
    - `file`: UploadFile object representing the CSV file containing the data for prediction.
    - `id_column`: Optional string representing the name of the ID column in the CSV file. If not provided, the default ID column will be used.
    - `response_format`: String representing the format in which the predictions should be returned. Valid values are `json` and `csv`.
    - `include_confidence`: Boolean representing whether to include confidence scores in the predictions.

    **Returns**:
    - If response_format is `json`, returns a `JSONResponse` object containing the predictions as a dictionary.
    - If response_format is `csv`, returns a `FileResponse` object containing the predictions as a CSV file.

    **Raises**:
    - `HTTPException` with status code 400 if:
        - An error occurs while reading the uploaded file
        - The ID column is not found.
        - One or more feature columns are missing.
        - The type of the features in the uploaded file doesn't match the expected types.

    - `HTTPException` with status code 500 if an error occurs while making predictions.
    """
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"An error occurred while reading the uploaded file: {e}",
        )

    id_column = id_column or settings.DEFAULT_ID_COLUMN

    features = settings.MODEL_FEATURES

    try:
        validate_data(df, id_column)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        model = get_model()
        predictions = model.predict(df[features])
        if include_confidence:
            confidence = model.predict_proba(df[features]).max(axis=1)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred while making predictions: {e}"
        )

    prediction_df = pd.DataFrame({"Prediction": predictions}, index=df[id_column])
    if include_confidence:
        prediction_df["Confidence"] = confidence

    if response_format == "json":
        return JSONResponse(prediction_df.to_dict(orient="index"))
    elif response_format == "csv":
        # This is to avoid race conditions when serving multiple requests.
        tmp_file = f"temp/{uuid4()}.csv"
        os.makedirs("temp", exist_ok=True)
        prediction_df.to_csv(tmp_file, index_label=id_column)
        return FileResponse(
            tmp_file,
            media_type="text/csv",
            filename="predictions.csv",
            background=BackgroundTask(os.remove, tmp_file),
        )


@app.get("/health")
async def health_check():
    """
    Check the health of the API.

    Returns:
    - The status of the API
    """
    return {"status": "ok"}
