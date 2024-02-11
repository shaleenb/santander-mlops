FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

COPY ./frontend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./frontend/app.py /app/

EXPOSE 8501

CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0
