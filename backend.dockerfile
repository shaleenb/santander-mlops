FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

COPY ./backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./backend /app/backend
COPY ./ml/feature_engineering.py /app
COPY ./ml/models/model.joblib /app/models/model.joblib

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
