import os
from io import StringIO

import pandas as pd
import requests
import streamlit as st

prediction_service_url = os.getenv("PREDICTION_SERVICE_URL")


st.title("Santander Customer Transaction Prediction")

st.write(
    "This is a simple demo of the Santander Customer Transaction Prediction competition on Kaggle."
)

file = st.file_uploader("Upload a CSV file", type=["csv"])
id_column = st.text_input("ID column", "ID_code")
response_format = st.selectbox("Response format", ["json", "csv"])
include_confidence = st.checkbox("Include confidence scores")
as_file = st.checkbox("Download as file")

if file:
    pred_button = st.button("Predict")
    if pred_button:
        response = requests.post(
            f"{prediction_service_url}/predict",
            files={"file": file},
            params={
                "id_column": id_column,
                "response_format": response_format,
                "include_confidence": include_confidence,
            },
            timeout=120,
        )
        if not response.ok:
            st.write("An error occurred with the request:")
            st.write(response.json()["detail"])
        else:
            if response_format == "json":
                if as_file:
                    st.download_button(
                        "Download JSON",
                        response.content,
                        file_name="predictions.json",
                        mime="application/json",
                    )
                else:
                    st.json(response.json())
            elif response_format == "csv":
                if as_file:
                    st.download_button(
                        "Download CSV",
                        response.content,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                else:
                    df = pd.read_csv(StringIO(response.text), index_col=id_column)
                    st.dataframe(df)
