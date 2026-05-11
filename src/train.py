# src/train.py
import json
import joblib
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

def build_final_pipeline(preprocessor):
    """
    Construye el pipeline completo:
    1. Preprocesamiento (encoding + scaling)
    2. Selección de features vía L1 (Logistic Regression con regularización Lasso)
    3. Regresión Logística final 
    
    Nota para no olvidarme: Importante evitar data leakage como me pasó por aplicar el scaler a X. Al usar Pipeline, el selector se ajusta SOLO con train durante .fit(),
    garantizando cero data leakage.
    """
    
    selector = SelectFromModel(
LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42))
    
    classifier = LogisticRegression(
        penalty="l1", solver="saga", C=0.5,
        class_weight={0: 1, 1: 1.6},  # Compensa desbalanceo hacia clase minoritaria {1.6 valor que mejora recall sin sacrificar tanto precision}
        max_iter=1000, random_state=42
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('classifier', classifier)
    ])

def train_evaluate_save(pipeline, X_train, X_test, y_train, y_test, model_dir='models'):
    """
    Entrena, evalúa y serializa el pipeline completo.
    Justificación de métricas:
    - Accuracy: referencia global, pero engañosa en datasets desbalanceados.
    - Precision/Recall/F1: foco en la clase positiva (Churn). Penaliza falsos positivos/negativos.
    - ROC-AUC: mide capacidad discriminativa independiente del threshold. Ideal para comparar modelos.
    - Matriz de confusión: visualiza trade-off entre clases.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Entrenamiento completo (preprocessor + selector + classifier)
    pipeline.fit(X_train, y_train)

    # Predicciones
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Cálculo de métricas
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_prob))
    }

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Guardar métricas en JSON
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Serializar artefacto completo (preprocesamiento + selección + modelo)
    model_path = os.path.join(model_dir, 'model_pipeline.joblib')
    joblib.dump(pipeline, model_path)

    # Reporte en consola
    print("\n" + "="*60)
    print("RESULTADOS DEL MODELO FINAL (Logistic Regression)")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"\n✓ Métricas guardadas en {metrics_path}")
    print(f"✓ Pipeline serializado en {model_path}")

    return metrics, cm