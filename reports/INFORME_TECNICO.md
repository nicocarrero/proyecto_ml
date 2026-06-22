# Informe Técnico — Predicción de Churn

**AndesLink Servicios Digitales S.A.**

---

# 1. Problema

AndesLink enfrenta un incremento sostenido en la tasa de cancelación voluntaria de clientes (*churn*), con impacto directo en los ingresos recurrentes, el costo de adquisición y la eficiencia de las campañas comerciales.

El problema se aborda como una **clasificación binaria supervisada**: dado un cliente, predecir si cancelará o no su suscripción.

**Variable objetivo:** `churn`

* `1` → el cliente canceló su suscripción.
* `0` → el cliente continúa activo.

El modelo estima probabilidades de churn, lo que habilita la segmentación por niveles de riesgo y la toma de decisiones basada en umbrales configurables.

---

# 2. Dataset

| Característica | Detalle                         |
| -------------- | ------------------------------- |
| Archivo        | `churn_sintetico.csv`           |
| Filas          | 5.000 registros                 |
| Columnas       | 16 variables originales         |
| Nulos          | Ninguno                         |
| Duplicados     | Ninguno                         |
| Balance        | Variable objetivo desbalanceada |

## Variables más relevantes (EDA)

* `contract_type`, `payment_method`, `internet_service`: alta incidencia en churn.
* `tenure_months`: principal factor de retención (correlación con churn ≈ -0.17).
* `support_tickets`: fuerte indicador de fuga (correlación con churn ≈ +0.10).
* `total_charges` y `tenure_months`: fuerte correlación entre sí (r ≈ 0.87).
* `region`: aporte limitado respecto a otras variables más relevantes.

---

# 3. Preparación de Datos

## 3.1 Transformaciones sobre variables existentes

### total_charges → categorización por cuartiles

La variable presenta asimetría y una fuerte correlación con `tenure_months`, por lo que se transformó en una variable categórica ordinal.

| Segmento   | Criterio    |
| ---------- | ----------- |
| Bajo       | ≤ Q1        |
| Medio-Bajo | Q1 < x ≤ Q2 |
| Medio-Alto | Q2 < x ≤ Q3 |
| Alto/VIP   | > Q3        |

La variable original fue eliminada.

### support_tickets → agrupación de valores extremos

Los valores mayores o iguales a cinco tickets fueron agrupados en una única categoría (`5+`) para reducir dispersión y capturar mejor el comportamiento observado.

La variable original fue eliminada.

---

## 3.2 Feature Engineering

Se generaron variables derivadas para capturar relaciones no lineales y comportamientos relevantes del cliente.

| Variable             | Descripción                                 |
| -------------------- | ------------------------------------------- |
| riesgo_contrato      | contract_type_mensual / (tenure_months + 1) |
| num_servicios        | Cantidad de servicios contratados           |
| cliente_problematico | Combina morosidad e incidencias             |
| anchor_score         | tenure_months × num_products                |

---

## 3.3 Preprocesamiento

| Paso                   | Decisión                                 |
| ---------------------- | ---------------------------------------- |
| Encoding ordinal       | total_charges_cat                        |
| Encoding nominal       | One-Hot Encoding                         |
| Escalado               | StandardScaler                           |
| División train/test    | 80/20 con estratificación                |
| Selección de variables | SelectFromModel + Logistic Regression L1 |

---

# 4. Modelos y Métricas

## 4.1 Baseline

Se evaluaron tres algoritmos principales.

| Modelo              | Resultado                            |
| ------------------- | ------------------------------------ |
| Logistic Regression | Mejor desempeño general              |
| Random Forest       | Inferior a Logistic Regression       |
| XGBoost             | Similar, sin ventajas significativas |

---

## 4.2 Optimización

Se utilizó RandomizedSearchCV con StratifiedKFold de 5 folds.

### Logistic Regression

* penalty = L1
* C = 0.0335

F1 promedio en validación cruzada:

```text
0.609
```

---

### Random Forest

No logró superar a Logistic Regression.

---

### XGBoost

Obtuvo resultados competitivos, aunque sin mejoras relevantes que justificaran la complejidad adicional.

---

## 4.3 Modelo Final

Configuración seleccionada:

```python
LogisticRegression(
    penalty="l1",
    solver="saga",
    C=0.5,
    class_weight={0:1, 1:1.6}
)
```

### Validación Cruzada

| Métrica   | Valor aproximado |
| --------- | ---------------- |
| F1        | 0.59             |
| ROC-AUC   | 0.76             |
| Precision | 0.55             |
| Recall    | 0.64             |

### Evaluación sobre Test Set

| Métrica   | Valor |
| --------- | ----- |
| Precision | 0.61  |
| Recall    | 0.70  |
| ROC-AUC   | 0.80  |

El modelo prioriza la detección de clientes con riesgo de abandono manteniendo una precisión razonable para campañas de retención.

---

# 5. Arquitectura de la Solución

La solución fue diseñada siguiendo una arquitectura desacoplada.

```text
Usuario
   │
   ▼
Streamlit
   │
   ▼
FastAPI
   │
   ▼
Modelo ML
```

Componentes principales:

* FastAPI para servir inferencias.
* Streamlit para interacción con usuarios.
* Joblib para serialización del modelo.
* Docker para contenedorización.
* Docker Compose para orquestación local.

---

# 6. Reproducibilidad y MLOps

Con el objetivo de garantizar trazabilidad y reproducibilidad se incorporaron herramientas MLOps.

| Herramienta    | Propósito                        |
| -------------- | -------------------------------- |
| Git            | Versionado de código             |
| DVC            | Versionado de datasets y modelos |
| MLflow         | Tracking de experimentos         |
| Docker         | Empaquetado de servicios         |
| Docker Compose | Despliegue reproducible          |
| Pytest         | Testing automatizado             |

## DVC

Los datasets y modelos se gestionan mediante DVC, evitando almacenar archivos pesados directamente en Git.

Artefactos versionados:

* Dataset original.
* Dataset procesado.
* Modelo final.

## MLflow

MLflow registra:

* Parámetros.
* Métricas.
* Artefactos.
* Matrices de confusión.
* Comparación entre experimentos.

---

# 7. Servicio de Inferencia

La API fue desarrollada utilizando FastAPI.

## Endpoint principal

```http
POST /predict
```

La API recibe los datos del cliente, ejecuta el pipeline completo y devuelve:

* Predicción.
* Probabilidad de churn.

La documentación interactiva se encuentra disponible mediante Swagger UI.

---

# 8. Interfaz Gráfica

Se desarrolló una interfaz con Streamlit para facilitar el uso de la solución.

Permite:

* Ingresar datos del cliente.
* Consumir la API de inferencia.
* Visualizar resultados.
* Mostrar probabilidades asociadas.

Esta capa desacopla al usuario final de la API y simplifica las pruebas funcionales.

---

# 9. Testing

Se implementaron pruebas automatizadas utilizando Pytest.

Cobertura principal:

* API.
* Preprocesamiento.
* Modelo.
* Validación de respuestas.

Las pruebas se ejecutan mediante:

```bash
pytest -v
```

---

# 10. Conclusiones y Trabajo Futuro

La Regresión Logística con regularización L1 y pesos de clase ajustados resultó ser la alternativa más adecuada para este problema.

Principales ventajas:

* Buen equilibrio entre precisión y recall.
* Alta interpretabilidad.
* Estabilidad en validación cruzada.
* Menor complejidad operativa.

## Insights de negocio

* Contrato mensual + pago en efectivo + internet móvil representan el perfil de mayor riesgo.
* A partir del tercer ticket de soporte aumenta significativamente la probabilidad de abandono.
* Los clientes Alto/VIP presentan la mayor estabilidad.
* Las estrategias de retención deberían focalizarse en clientes de bajo compromiso contractual.

## Trabajo Futuro

Como siguiente etapa del proyecto se incorporarán:

* Prometheus.
* Grafana.
* Evidently AI.
* Monitoreo de drift.
* Observabilidad de modelos.
* Dashboards operativos.

---

# 11. Limitaciones

* Dataset sintético.
* Desbalance natural de clases.
* Supuesto de linealidad de la Regresión Logística.
* Ausencia de variables temporales.
* Optimización de hiperparámetros no exhaustiva.
* Umbral de clasificación fijo en 0.5.

Estas limitaciones representan oportunidades de mejora para futuras iteraciones del sistema.

```
```
