import os
from uuid import uuid4

import joblib
import pandas as pd
from config import settings
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

app = FastAPI()

model = joblib.load(settings.MODEL_PATH)


# TODO: async this
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    id_column: str | None = None,
    response_format: str = "json",
):
    """
    Make predictions on the uploaded CSV file using the trained model.

    Args:
        file (UploadFile): The uploaded CSV file.
        id_column (str, optional): The name of the ID column. Defaults to None.
        response_format (str, optional): The format of the response. Defaults to "json".
            Allowed values: "json", "csv".

    Returns:
        JSONResponse | FileResponse: The predictions in the specified format.

    Raises:
        HTTPException: If an error occurs while reading the uploaded file or making predictions.
    """
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"An error occurred while reading the uploaded file: {e}",
        )

    id_column = id_column or settings.DEFAULT_ID_COLUMN

    if id_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"ID column '{id_column}' not found in the uploaded file.",
        )

    features = [col for col in df.columns if col != id_column]

    try:
        predictions = model.predict(df[features])
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred while making predictions: {e}"
        )

    prediction_series = pd.Series(predictions, index=df[id_column])
    prediction_series.index.name = id_column
    prediction_series.name = "Prediction"

    if response_format == "json":
        return JSONResponse(prediction_series.to_dict())
    elif response_format == "csv":
        # This is to avoid race conditions when serving multiple requests.
        tmp_file = f"temp/{uuid4()}.csv"
        os.makedirs("temp", exist_ok=True)
        prediction_series.to_csv(tmp_file, index_label=id_column)
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
        dict: The status of the API.
    """
    return {"status": "ok"}
