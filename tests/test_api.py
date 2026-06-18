# -*- coding: utf-8 -*-
"""
tests/test_api.py
-----------------
Suite de pruebas para la API de inferencia de churn (AndesLink).
Cubre: health check, predicción individual, predicción batch,
validación de esquemas, lógica de risk_label y manejo de errores.

Funciona con MLflow o joblib — parchea la carga del modelo antes
de importar la app para evitar que falle al inicializar.

Ejecutar con:
    pytest tests/test_api.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import numpy as np

# ---------------------------------------------------------------------------
# Fixture: cliente de prueba con modelo mockeado
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """
    Crea un TestClient con el modelo reemplazado por un mock que siempre
    predice churn=1 con probabilidad 0.85 (risk_level='alto').

    Parchea tanto mlflow.sklearn.load_model como joblib.load ANTES de
    importar la app, para que funcione sin importar cómo cargue la API.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])

    # Parchear ambas posibles fuentes de carga antes del import
    with patch("mlflow.sklearn.load_model", return_value=mock_model), patch(
        "joblib.load", return_value=mock_model
    ):

        from src.api.api import app

        app.state.model = mock_model  # por si la app guarda referencia en state
        with TestClient(app) as c:
            yield c


@pytest.fixture
def valid_customer():
    """Payload válido de un cliente representativo."""
    return {
        "tenure_months": 12,
        "monthly_charge": 65.50,
        "total_charges": 786.0,
        "support_tickets": 1,
        "late_payments": 0,
        "avg_monthly_usage_gb": 95.3,
        "contract_type": "anual",
        "payment_method": "debito",
        "internet_service": "fibra",
        "has_streaming": 1,
        "has_security_pack": 0,
        "num_products": 2,
        "region": "centro",
        "customer_age": 35,
        "is_promo": 0,
    }


# ===========================================================================
# 1. Health check
# ===========================================================================
class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body_has_status_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_model_loaded_true(self, client):
        """Con el mock activo, model_loaded debe ser True."""
        body = client.get("/health").json()
        assert body["model_loaded"] is True

    def test_health_returns_model_uri(self, client):
        """El health devuelve model_uri (MLflow Model Registry) en lugar de model_path."""
        body = client.get("/health").json()
        assert "model_uri" in body

    def test_health_model_uri_contains_model_name(self, client):
        body = client.get("/health").json()
        uri = body.get("model_uri", "")
        assert "models:/" in uri or len(uri) > 0


# ===========================================================================
# 2. POST /predict — casos felices
# ===========================================================================
class TestPredictHappyPath:
    def test_predict_returns_200(self, client, valid_customer):
        response = client.post("/predict", json=valid_customer)
        assert response.status_code == 200

    def test_predict_response_has_required_fields(self, client, valid_customer):
        body = client.post("/predict", json=valid_customer).json()
        assert "churn_prediction" in body
        assert "churn_probability" in body
        assert "risk_level" in body

    def test_predict_churn_prediction_is_binary(self, client, valid_customer):
        body = client.post("/predict", json=valid_customer).json()
        assert body["churn_prediction"] in (0, 1)

    def test_predict_probability_between_0_and_1(self, client, valid_customer):
        body = client.post("/predict", json=valid_customer).json()
        assert 0.0 <= body["churn_probability"] <= 1.0

    def test_predict_risk_level_valid_value(self, client, valid_customer):
        body = client.post("/predict", json=valid_customer).json()
        assert body["risk_level"] in ("bajo", "medio", "alto")

    def test_predict_contract_mensual(self, client, valid_customer):
        payload = {**valid_customer, "contract_type": "mensual"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_contract_bianual(self, client, valid_customer):
        payload = {**valid_customer, "contract_type": "bianual"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_all_payment_methods(self, client, valid_customer):
        for method in ("transferencia", "debito", "efectivo", "credito"):
            payload = {**valid_customer, "payment_method": method}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, f"Falló con payment_method='{method}'"

    def test_predict_all_internet_services(self, client, valid_customer):
        for service in ("cable", "fibra", "movil", "ninguno"):
            payload = {**valid_customer, "internet_service": service}
            response = client.post("/predict", json=payload)
            assert (
                response.status_code == 200
            ), f"Falló con internet_service='{service}'"

    def test_predict_all_regions(self, client, valid_customer):
        for region in ("centro", "norte", "oeste", "sur"):
            payload = {**valid_customer, "region": region}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, f"Falló con region='{region}'"

    def test_predict_boundary_tenure_min(self, client, valid_customer):
        """tenure_months en su valor mínimo permitido (1)."""
        payload = {**valid_customer, "tenure_months": 1}
        assert client.post("/predict", json=payload).status_code == 200

    def test_predict_boundary_tenure_max(self, client, valid_customer):
        """tenure_months en su valor máximo permitido (72)."""
        payload = {**valid_customer, "tenure_months": 72}
        assert client.post("/predict", json=payload).status_code == 200

    def test_predict_boundary_customer_age_min(self, client, valid_customer):
        payload = {**valid_customer, "customer_age": 18}
        assert client.post("/predict", json=payload).status_code == 200

    def test_predict_boundary_customer_age_max(self, client, valid_customer):
        payload = {**valid_customer, "customer_age": 78}
        assert client.post("/predict", json=payload).status_code == 200


# ===========================================================================
# 3. POST /predict — validación de esquema (422)
# ===========================================================================
class TestPredictValidation:
    def test_missing_required_field_returns_422(self, client, valid_customer):
        payload = {k: v for k, v in valid_customer.items() if k != "tenure_months"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_tenure_below_minimum_returns_422(self, client, valid_customer):
        payload = {**valid_customer, "tenure_months": 0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_tenure_above_maximum_returns_422(self, client, valid_customer):
        payload = {**valid_customer, "tenure_months": 73}
        assert client.post("/predict", json=payload).status_code == 422

    def test_monthly_charge_below_minimum_returns_422(self, client, valid_customer):
        payload = {**valid_customer, "monthly_charge": 10.0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_monthly_charge_above_maximum_returns_422(self, client, valid_customer):
        payload = {**valid_customer, "monthly_charge": 200.0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_invalid_contract_type_returns_422(self, client, valid_customer):
        payload = {**valid_customer, "contract_type": "semestral"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_invalid_payment_method_returns_422(self, client, valid_customer):
        payload = {**valid_customer, "payment_method": "criptomoneda"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_invalid_internet_service_returns_422(self, client, valid_customer):
        payload = {**valid_customer, "internet_service": "satelite"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_invalid_region_returns_422(self, client, valid_customer):
        payload = {**valid_customer, "region": "patagonia"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_has_streaming_invalid_value_returns_422(self, client, valid_customer):
        payload = {**valid_customer, "has_streaming": 2}
        assert client.post("/predict", json=payload).status_code == 422

    def test_support_tickets_above_max_returns_422(self, client, valid_customer):
        payload = {**valid_customer, "support_tickets": 9}
        assert client.post("/predict", json=payload).status_code == 422

    def test_late_payments_above_max_returns_422(self, client, valid_customer):
        payload = {**valid_customer, "late_payments": 6}
        assert client.post("/predict", json=payload).status_code == 422

    def test_num_products_above_max_returns_422(self, client, valid_customer):
        payload = {**valid_customer, "num_products": 5}
        assert client.post("/predict", json=payload).status_code == 422

    def test_customer_age_below_min_returns_422(self, client, valid_customer):
        payload = {**valid_customer, "customer_age": 17}
        assert client.post("/predict", json=payload).status_code == 422

    def test_empty_body_returns_422(self, client):
        assert client.post("/predict", json={}).status_code == 422


# ===========================================================================
# 4. POST /predict/batch
# ===========================================================================
class TestPredictBatch:
    def test_batch_single_customer_returns_200(self, client, valid_customer):
        payload = {"customers": [valid_customer]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_multiple_customers_returns_200(self, client, valid_customer):
        payload = {"customers": [valid_customer, valid_customer, valid_customer]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_total_matches_input_count(self, client, valid_customer):
        n = 5
        payload = {"customers": [valid_customer] * n}
        body = client.post("/predict/batch", json=payload).json()
        assert body["total"] == n

    def test_batch_predictions_list_length_matches_total(self, client, valid_customer):
        n = 3
        payload = {"customers": [valid_customer] * n}
        body = client.post("/predict/batch", json=payload).json()
        assert len(body["predictions"]) == body["total"]

    def test_batch_churn_count_lte_total(self, client, valid_customer):
        payload = {"customers": [valid_customer] * 4}
        body = client.post("/predict/batch", json=payload).json()
        assert body["churn_count"] <= body["total"]

    def test_batch_churn_rate_between_0_and_1(self, client, valid_customer):
        payload = {"customers": [valid_customer] * 4}
        body = client.post("/predict/batch", json=payload).json()
        assert 0.0 <= body["churn_rate"] <= 1.0

    def test_batch_churn_rate_equals_churn_count_over_total(
        self, client, valid_customer
    ):
        payload = {"customers": [valid_customer] * 4}
        body = client.post("/predict/batch", json=payload).json()
        expected_rate = round(body["churn_count"] / body["total"], 4)
        assert body["churn_rate"] == expected_rate

    def test_batch_empty_list_returns_422(self, client):
        """La lista vacía debe rechazarse (min_length=1)."""
        payload = {"customers": []}
        assert client.post("/predict/batch", json=payload).status_code == 422

    def test_batch_missing_customers_key_returns_422(self, client):
        assert client.post("/predict/batch", json={}).status_code == 422

    def test_batch_each_prediction_has_required_fields(self, client, valid_customer):
        payload = {"customers": [valid_customer]}
        predictions = client.post("/predict/batch", json=payload).json()["predictions"]
        for pred in predictions:
            assert "churn_prediction" in pred
            assert "churn_probability" in pred
            assert "risk_level" in pred


# ===========================================================================
# 5. GET /schema
# ===========================================================================
class TestSchema:
    def test_schema_returns_200(self, client):
        assert client.get("/schema").status_code == 200

    def test_schema_contains_properties(self, client):
        body = client.get("/schema").json()
        assert "properties" in body

    def test_schema_has_all_expected_fields(self, client):
        expected_fields = {
            "tenure_months",
            "monthly_charge",
            "total_charges",
            "support_tickets",
            "late_payments",
            "avg_monthly_usage_gb",
            "contract_type",
            "payment_method",
            "internet_service",
            "has_streaming",
            "has_security_pack",
            "num_products",
            "region",
            "customer_age",
            "is_promo",
        }
        properties = set(client.get("/schema").json()["properties"].keys())
        assert expected_fields == properties


# ===========================================================================
# 6. Lógica de risk_label (unit test puro, sin HTTP)
# ===========================================================================
class TestRiskLabel:
    """
    Prueba la función risk_label directamente, sin levantar la API.
    Importación separada para aislar la lógica de negocio.
    """

    @pytest.fixture(autouse=True)
    def import_risk_label(self):
        with patch("mlflow.sklearn.load_model", return_value=MagicMock()), patch(
            "joblib.load", return_value=MagicMock()
        ):
            from src.api.api import risk_label

            self.risk_label = risk_label

    def test_probability_0_is_bajo(self):
        assert self.risk_label(0.0) == "bajo"

    def test_probability_034_is_bajo(self):
        assert self.risk_label(0.34) == "bajo"

    def test_probability_035_is_medio(self):
        assert self.risk_label(0.35) == "medio"

    def test_probability_064_is_medio(self):
        assert self.risk_label(0.64) == "medio"

    def test_probability_065_is_alto(self):
        assert self.risk_label(0.65) == "alto"

    def test_probability_1_is_alto(self):
        assert self.risk_label(1.0) == "alto"

    def test_boundary_exact_035(self):
        """0.35 es el límite inferior de 'medio'."""
        assert self.risk_label(0.35) == "medio"

    def test_boundary_exact_065(self):
        """0.65 es el límite inferior de 'alto'."""
        assert self.risk_label(0.65) == "alto"


# ===========================================================================
# 7. Manejo de modelo no disponible (503)
# ===========================================================================
class TestModelUnavailable:
    """
    Simula el escenario donde el modelo no pudo cargarse (model = None).
    """

    @pytest.fixture
    def client_no_model(self):
        with patch("mlflow.sklearn.load_model", side_effect=Exception), patch(
            "joblib.load", side_effect=FileNotFoundError
        ):

            from src.api import api as api_module

            original_model = api_module.model
            api_module.model = None
            api_module.app.state.model = None

            with TestClient(api_module.app) as c:
                yield c

            api_module.model = original_model
            api_module.app.state.model = original_model

    def test_predict_without_model_returns_503(self, client_no_model, valid_customer):
        response = client_no_model.post("/predict", json=valid_customer)
        assert response.status_code == 503

    def test_predict_batch_without_model_returns_503(
        self, client_no_model, valid_customer
    ):
        payload = {"customers": [valid_customer]}
        response = client_no_model.post("/predict/batch", json=payload)
        assert response.status_code == 503

    def test_health_without_model_shows_model_loaded_false(self, client_no_model):
        body = client_no_model.get("/health").json()
        assert body["model_loaded"] is False
