# -*- coding: utf-8 -*-
"""debug_mlflow.py — Ejecutar esto para ver exactamente dónde falla."""

from dotenv import load_dotenv
import os
from pathlib import Path

# 1. Encontrar .env explícitamente (funciona desde cualquier carpeta)
env_path = Path(__file__).parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).parent.parent / ".env"

print(f"Buscando .env en: {env_path.absolute()}")
print(f"Existe: {env_path.exists()}")

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print("✅ .env cargado")
else:
    print("❌ .env no encontrado")

# 2. Verificar variables
user = os.getenv("MLFLOW_TRACKING_USERNAME")
pw = os.getenv("MLFLOW_TRACKING_PASSWORD")
uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/carreronicoo/proyecto_ml.mlflow")

print(f"Variables:")
print(f"  USER: {'***' if user else 'NO ESTÁ (None)'}")
print(f"  PASS: {'***' if pw else 'NO ESTÁ (None)'}")
print(f"  URI:  {uri}")

if not user or not pw:
    print("❌ Faltan credenciales. Verificá que .env tenga:")
    print("   MLFLOW_TRACKING_USERNAME=carreronicoo")
    print("   MLFLOW_TRACKING_PASSWORD=tu_token")
    exit(1)

# 3. Probar conexión a DagsHub
print("🔌 Probando conexión a MLflow...")
try:
    import mlflow
    mlflow.set_tracking_uri(uri)

    # Listar experimentos
    exp = mlflow.get_experiment_by_name("Default")
    print(f"✅ Conexión OK. Experimento Default: {exp}")

    # Listar modelos registrados
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    print("📋 Modelos registrados:")
    for rm in client.search_registered_models():
        print(f"   - {rm.name}")

    # Intentar cargar el modelo
    model_name = "CustomerChurn"  # Cambiá esto si tu modelo tiene otro nombre
    print(f"🎯 Buscando modelo: {model_name}")

    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            print(f"✅ Versión en Production: {versions[0].version}")
        else:
            print("⚠️ No hay versión en stage 'Production'")
            print("   Buscando en stage 'None' (latest)...")
            versions = client.get_latest_versions(model_name, stages=["None"])
            if versions:
                print(f"   Encontrado en 'None': v{versions[0].version}")
    except Exception as e:
        print(f"❌ Error buscando modelo: {e}")
        print("💡 Posibles causas:")
        print("   - El nombre del modelo no es 'CustomerChurn'")
        print("   - No hay modelo registrado en DagsHub")
        print("   - El token no tiene permisos")

except Exception as e:
    print(f"❌ Error de conexión: {e}")
