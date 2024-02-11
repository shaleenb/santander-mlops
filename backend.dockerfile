FROM --platform=linux/amd64 python:3.11-slim

ARG model

WORKDIR /app

COPY ./backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./backend/api.py /app/
COPY ./ml/models/${model} /app/models/model.joblib

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
