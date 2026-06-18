# -*- coding: utf-8 -*-
"""
api.py
------
API de inferencia de churn. Carga model_pipeline.joblib y expone
endpoints de prediccion usando los esquemas definidos en schemas.py.
"""

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os
import mlflow
import sys
from pathlib import Path

from .schemas import (
    CustomerFeatures,
    BatchRequest,
    PredictionResponse,
    BatchPredictionResponse,
)



os.environ["MLFLOW_TRACKING_USERNAME"] = "carreronicoo"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "11c0bc53ab66e42295a8f9c55704bfec4c3580c0"

mlflow.set_tracking_uri(
    "https://dagshub.com/carreronicoo/proyecto_ml.mlflow"
)
# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Churn Prediction API",
    description="API de inferencia para prediccion de churn de clientes.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------
MODEL_URI = "models:/CustomerChurn/latest"
try:
    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    print("[OK] Modelo cargado desde MLflow Registry")

except Exception as e:
    model = None
    print(f"[ERROR] {e}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "tenure_months", "monthly_charge", "total_charges", "support_tickets",
    "late_payments", "avg_monthly_usage_gb", "contract_type", "payment_method",
    "internet_service", "has_streaming", "has_security_pack", "num_products",
    "region", "customer_age", "is_promo",
]


def risk_label(prob: float) -> str:
    if prob < 0.35:
        return "bajo"
    elif prob < 0.65:
        return "medio"
    return "alto"


def predict_one(customer: CustomerFeatures) -> PredictionResponse:
    df = pd.DataFrame([customer.model_dump()])[FEATURE_ORDER]
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return PredictionResponse(
        churn_prediction=pred,
        churn_probability=round(prob, 4),
        risk_level=risk_label(prob),
    )


def check_model():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible. Verifica acceso a '{MODEL_URI}'.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
def health():
    """Verifica que la API y el modelo esten disponibles."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediccion"])
def predict(customer: CustomerFeatures):
    """
    Predice si un solo cliente va a hacer churn.

    - churn_prediction: 0 (no churn) o 1 (churn).
    - churn_probability: probabilidad entre 0 y 1.
    - risk_level: bajo (<35%) | medio (35-65%) | alto (>65%).
    """
    check_model()
    return predict_one(customer)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediccion"])
def predict_batch(request: BatchRequest):
    """
    Predice churn para multiples clientes en una sola llamada (max. 1000).
    """
    check_model()
    results = [predict_one(c) for c in request.customers]
    churn_count = sum(r.churn_prediction for r in results)
    return BatchPredictionResponse(
        predictions=results,
        total=len(results),
        churn_count=churn_count,
        churn_rate=round(churn_count / len(results), 4),
    )


@app.get("/schema", tags=["utilidades"])
def schema():
    """Devuelve el JSON Schema de los campos de entrada esperados."""
    return CustomerFeatures.model_json_schema()