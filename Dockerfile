# -------------------------------------------------------
# Dockerfile — API de inferencia (FastAPI)
# -------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copiar dependencias primero (mejor uso de caché de Docker)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copiar el código fuente
COPY src/ ./src/

# Variables de entorno para MLflow / DagsHub
# Se pasan en docker-compose.yml o como secrets en CI
ENV MLFLOW_TRACKING_USERNAME=""
ENV MLFLOW_TRACKING_PASSWORD=""

EXPOSE 8000

CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
