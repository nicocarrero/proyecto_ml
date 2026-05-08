# src/main.py
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Aseguramos que podamos importar desde la raíz del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importamos las funciones de tus otros módulos
from src.feature_engineering import run_feature_engineering
from src.preprocessing import split_and_preprocess
from src.train import build_final_pipeline, train_evaluate_save

def main():
    print("- Iniciando pipeline de entrenamiento integral...")

    # --- PASO 1: Cargar Datos Crudos (Raw) ---
    raw_path = 'data/raw/churn_sintetico.csv'
    processed_path = 'data/processed/data_final.csv'
    
    try:
        print(f"- Cargando datos desde {raw_path}...")
        df_raw = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en {raw_path}.")
        return

    # --- PASO 2: Ejecutar Feature Engineering ---
    print("- Ejecutando Feature Engineering...")
    try:
        df_processed = run_feature_engineering(df_raw)
        
        # Guardamos el resultado del feature engineering para transparencia y reproducibilidad
        os.makedirs('data/processed', exist_ok=True)
        df_processed.to_csv(processed_path, index=False)
        print(f"✓ Feature Engineering completado. Guardado en: {processed_path}")
    except Exception as e:
        print(f"❌ Error en Feature Engineering: {e}")
        return

    # --- PASO 3: Split + Preprocessing---
    print("- Procesando datos (Split, Encoding, Scaling)...")
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(df_processed)

    # --- PASO 4: Construir pipeline completo ---
    print("- Construyendo pipeline final...")
    pipeline = build_final_pipeline(preprocessor)

    # --- PASO 5: Entrenar, evaluar y serializar ---
    print("- Entrenando y evaluando modelo...")
    metrics, cm = train_evaluate_save(pipeline, X_train, X_test, y_train, y_test)

    # --- PASO 6: Generar evidencia gráfica ---
    print("- Generando evidencia gráfica...")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Matriz de Confusión - Modelo Final (Logistic Regression)')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/confusion_matrix.png', dpi=300)
    plt.close()
    
    print("\n" + "="*50)
    print("¡PIPELINE EJECUTADO EXITOSAMENTE!")
    print("="*50)
    print(f"Artefactos actualizados en 'models/' y 'data/processed/'")

if __name__ == '__main__':
    main()