# -*- coding: utf-8 -*-
"""
tests/test_modelo.py
--------------------
Tests del modelo productivo cargado desde MLflow Model Registry (DagsHub)
y tests de estructura del pipeline.

Las credenciales MLFLOW_TRACKING_USERNAME y MLFLOW_TRACKING_PASSWORD
son obligatorias. Se leen de variables de entorno (nunca hardcodeadas).

En GitHub Actions: configurarlas como Repository Secrets.
En local: exportarlas antes de correr pytest.

Ejecutar localmente:
    export MLFLOW_TRACKING_USERNAME=carreronicoo
    export MLFLOW_TRACKING_PASSWORD=tu_token
    pytest tests/test_modelo.py -v

Ejecutar solo tests de pipeline (sin DagsHub):
    pytest tests/test_modelo.py -v -k "Pipeline"
"""

from dotenv import load_dotenv

load_dotenv()  # Lee .env si existe, si no, sigue con las variables del sistema

import os

import numpy as np
import pandas as pd
import pytest

DAGSHUB_USER = "carreronicoo"
DAGSHUB_URI = f"https://dagshub.com/{DAGSHUB_USER}/proyecto_ml.mlflow"
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "CustomerChurn")
MODEL_URI = f"models:/{MODEL_NAME}/latest"

# ---------------------------------------------------------------------------
# Fixture: carga el modelo desde DagsHub una sola vez por sesión
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def modelo_registry():
    """
    Carga el modelo desde MLflow Model Registry en DagsHub.
    Falla explícitamente si las credenciales no están configuradas.
    """
    import mlflow

    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not username or not password:
        pytest.fail(
            "Credenciales de MLflow no configuradas.\n"
            "Definir MLFLOW_TRACKING_USERNAME y MLFLOW_TRACKING_PASSWORD "
            "como variables de entorno o secrets de GitHub Actions."
        )

    mlflow.set_tracking_uri(DAGSHUB_URI)
    # mlflow lee automáticamente MLFLOW_TRACKING_USERNAME/PASSWORD del entorno
    return mlflow.sklearn.load_model(MODEL_URI)


# ---------------------------------------------------------------------------
# Fixtures: perfiles de negocio
# ---------------------------------------------------------------------------


@pytest.fixture
def perfil_churn_alto():
    """
    Cliente con alto riesgo de abandono:
    contrato mensual, 2 meses de tenure, 5 pagos tardíos, 7 tickets.
    """
    return pd.DataFrame(
        [
            {
                "tenure_months": 2,
                "monthly_charge": 120.0,
                "total_charges": 240.0,
                "support_tickets": 7,
                "late_payments": 5,
                "avg_monthly_usage_gb": 10.0,
                "contract_type": "mensual",
                "payment_method": "efectivo",
                "internet_service": "movil",
                "has_streaming": 0,
                "has_security_pack": 0,
                "num_products": 1,
                "region": "sur",
                "customer_age": 22,
                "is_promo": 0,
            }
        ]
    )


@pytest.fixture
def perfil_churn_bajo():
    """
    Cliente con bajo riesgo de abandono:
    contrato bianual, 60 meses de tenure, sin problemas, 4 productos.
    """
    return pd.DataFrame(
        [
            {
                "tenure_months": 60,
                "monthly_charge": 40.0,
                "total_charges": 2400.0,
                "support_tickets": 0,
                "late_payments": 0,
                "avg_monthly_usage_gb": 150.0,
                "contract_type": "bianual",
                "payment_method": "debito",
                "internet_service": "fibra",
                "has_streaming": 1,
                "has_security_pack": 1,
                "num_products": 4,
                "region": "centro",
                "customer_age": 45,
                "is_promo": 1,
            }
        ]
    )


# ===========================================================================
# 1. Tests de negocio
# ===========================================================================
class TestPerfilesDeNegocio:

    def test_perfil_churn_alto_supera_05(self, modelo_registry, perfil_churn_alto):
        """
        Cliente mensual, 2 meses, 5 pagos tardíos, 7 tickets → churn > 0.5.
        Si falla: el modelo no aprendió que estos son señales de abandono.
        """
        prob = modelo_registry.predict_proba(perfil_churn_alto)[0][1]
        assert prob > 0.5, (
            f"Perfil de alto riesgo obtuvo probabilidad {prob:.3f}. "
            "Revisar entrenamiento o feature engineering."
        )

    def test_perfil_churn_bajo_por_debajo_05(self, modelo_registry, perfil_churn_bajo):
        """
        Cliente bianual, 60 meses, sin problemas, 4 productos → churn < 0.5.
        Si falla: el modelo no discrimina clientes leales de los que se van.
        """
        prob = modelo_registry.predict_proba(perfil_churn_bajo)[0][1]
        assert prob < 0.5, (
            f"Perfil de bajo riesgo obtuvo probabilidad {prob:.3f}. "
            "Revisar entrenamiento o feature engineering."
        )

    def test_diferencia_entre_perfiles_es_significativa(
        self, modelo_registry, perfil_churn_alto, perfil_churn_bajo
    ):
        """
        La diferencia de probabilidad entre perfil alto y bajo
        debe ser al menos 0.2. Si no, el modelo no discrimina bien.
        """
        prob_alto = modelo_registry.predict_proba(perfil_churn_alto)[0][1]
        prob_bajo = modelo_registry.predict_proba(perfil_churn_bajo)[0][1]
        diff = prob_alto - prob_bajo
        assert diff >= 0.2, (
            f"Diferencia insuficiente: alto={prob_alto:.3f}, bajo={prob_bajo:.3f} "
            f"(diff={diff:.3f}). El modelo no discrimina bien entre extremos."
        )


# ===========================================================================
# 2. Integridad del artefacto
# ===========================================================================
class TestIntegridadDelModelo:

    def test_modelo_carga_desde_registry(self, modelo_registry):
        assert modelo_registry is not None

    def test_predict_retorna_binario(self, modelo_registry, perfil_churn_alto):
        preds = modelo_registry.predict(perfil_churn_alto)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_tiene_dos_columnas(self, modelo_registry, perfil_churn_alto):
        assert modelo_registry.predict_proba(perfil_churn_alto).shape[1] == 2

    def test_probabilidades_suman_1(self, modelo_registry, perfil_churn_alto):
        proba = modelo_registry.predict_proba(perfil_churn_alto)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_acepta_las_columnas_de_la_api(self, modelo_registry):
        """
        El modelo debe aceptar las mismas 15 columnas que la API le envía.
        Si hay desajuste entre schemas.py y el pipeline, falla aquí
        antes de que falle en producción.
        """
        sample = pd.DataFrame(
            [
                {
                    "tenure_months": 12,
                    "monthly_charge": 65.5,
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
            ]
        )
        try:
            modelo_registry.predict(sample)
        except Exception as e:
            pytest.fail(
                f"El modelo rechazó las columnas del schema de la API: {e}\n"
                "Verificar que schemas.py y el pipeline estén sincronizados."
            )


# ===========================================================================
# 3. Estructura del pipeline
# No requiere DagsHub.
# ===========================================================================
class TestEstructuraPipeline:

    @pytest.fixture(scope="class")
    def preprocessor(self):
        from src.preprocessing import get_preprocessor

        return get_preprocessor()

    def test_tiene_al_menos_tres_pasos(self, preprocessor):
        from src.train import build_final_pipeline

        pipeline = build_final_pipeline(preprocessor)
        assert len(pipeline.steps) >= 3

    def test_feature_engineering_es_primer_paso(self, preprocessor):
        """
        Feature engineering DEBE ir antes del preprocessor.
        El preprocessor espera columnas como total_charges_cat que
        solo existen después de que el FE las crea.
        """
        from src.train import build_final_pipeline

        pipeline = build_final_pipeline(preprocessor)
        assert pipeline.steps[0][0] == "feature_engineering", (
            f"Primer paso es '{pipeline.steps[0][0]}', debería ser 'feature_engineering'. "
            "El preprocessor va a fallar al buscar columnas que el FE todavía no creó."
        )

    def test_classifier_es_ultimo_paso(self, preprocessor):
        from src.train import build_final_pipeline

        pipeline = build_final_pipeline(preprocessor)
        assert pipeline.steps[-1][0] == "classifier"

    def test_clasificador_por_defecto_es_logistic_regression(self, preprocessor):
        from sklearn.linear_model import LogisticRegression

        from src.train import build_final_pipeline

        pipeline = build_final_pipeline(preprocessor)
        assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)

    def test_clasificador_custom_se_inyecta_correctamente(self, preprocessor):
        from sklearn.ensemble import RandomForestClassifier

        from src.train import build_final_pipeline

        clf = RandomForestClassifier(n_estimators=5, random_state=0)
        pipeline = build_final_pipeline(preprocessor, classifier=clf)
        assert isinstance(pipeline.named_steps["classifier"], RandomForestClassifier)
