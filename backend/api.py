import os

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

app = FastAPI()

# TODO: These go to a config file
model_path = "models/model.joblib"
model = joblib.load(model_path)

default_id_column = "ID_code"


# TODO: async this
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    id_column: str | None = None,
    response_format: str = "json",
):
    df = pd.read_csv(file.file)
    id_column = id_column or default_id_column
    features = [col for col in df.columns if col not in [id_column, "target"]]
    predictions = model.predict(df[features])
    prediction_series = pd.Series(predictions, index=df[id_column])
    prediction_series.index.name = id_column
    prediction_series.name = "Prediction"

    if response_format == "json":
        return prediction_series.to_dict()
    elif response_format == "csv":
        tmp_file = "tmp.csv"
        prediction_series.to_csv(tmp_file)
        return FileResponse(
            tmp_file,
            media_type="text/csv",
            filename="predictions.csv",
            background=BackgroundTask(os.remove, tmp_file),
        )
