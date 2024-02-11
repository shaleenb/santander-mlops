import argparse
import sys
from io import StringIO

import pandas as pd
import requests
import streamlit as st


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-service-url")
    return parser.parse_args(args)


args = parse_args(sys.argv[1:])

prediction_service_url = args.prediction_service_url


st.title("Santander Customer Transaction Prediction")

st.write(
    "This is a simple demo of the Santander Customer Transaction Prediction competition on Kaggle."
)

file = st.file_uploader("Upload a CSV file", type=["csv"])
id_column = st.text_input("ID column", "ID_code")
response_format = st.selectbox("Response format", ["json", "csv"])
as_file = st.checkbox("Download as file")

if file:
    pred_button = st.button("Predict")
    if pred_button:
        response = requests.post(
            f"{prediction_service_url}/predict",
            files={"file": file},
            params={"id_column": id_column, "response_format": response_format},
        )
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
                st.table(df)
