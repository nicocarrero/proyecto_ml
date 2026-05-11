# Informe Técnico — Predicción de Churn
**AndesLink Servicios Digitales S.A.**

---

## 1. Problema

AndesLink enfrenta un incremento sostenido en la tasa de cancelación voluntaria de clientes (churn), con impacto directo en los ingresos recurrentes, el costo de adquisición y la eficiencia de las campañas comerciales.

El problema se aborda como una **clasificación binaria supervisada**: dado un cliente, predecir si cancelará o no su suscripción.

**Variable objetivo:** `churn`
- `1` — el cliente canceló su suscripción
- `0` — el cliente continúa activo

El modelo estima probabilidades de churn, lo que habilita la segmentación por niveles de riesgo y la toma de decisiones basada en umbrales configurables.

---

## 2. Dataset

| Característica | Detalle |
|---|---|
| Archivo | `churn_sintetico.csv` |
| Filas | 5.000 registros |
| Columnas | 16 variables originales |
| Nulos | Ninguno |
| Duplicados | Ninguno |
| Balance | Variable objetivo desbalanceada (comportamiento esperado en churn) |

### Variables más relevantes (EDA)

- `contract_type`, `payment_method`, `internet_service`: alta incidencia en churn. El riesgo se dispara en contratos mensuales, servicios de internet móvil y pagos en efectivo.
- `tenure_months`: principal factor de retención (correlación con churn: -0.17).
- `support_tickets`: mayor indicador de fuga (correlación con churn: +0.10).
- `total_charges` y `tenure_months`: fuerte correlación entre sí (r = 0.87), lo que indica redundancia.
- `region`: sin diferencia significativa entre categorías; no es un factor determinante.

---

## 3. Decisiones de Preparación de Datos

### 3.1 Transformaciones sobre variables existentes

**`total_charges` → categorización por cuartiles**

La variable presenta asimetría extrema y es casi un proxy de `tenure_months` (r = 0.87). Se optó por categorizarla en cuatro segmentos basados en cuartiles:

| Segmento | Criterio |
|---|---|
| Bajo | ≤ Q1 |
| Medio-Bajo | Q1 < x ≤ Q2 |
| Medio-Alto | Q2 < x ≤ Q3 |
| Alto/VIP | > Q3 |

Resultado: a mayor gasto acumulado, menor tasa de churn. La variable numérica original fue eliminada del dataset.

**`support_tickets` → agrupación de valores extremos**

Se identificó un umbral crítico: al alcanzar 4 tickets de soporte, la tasa de abandono se duplica respecto a clientes sin incidencias. Los valores ≥ 5 se colapsaron en una categoría única (`5+`). La variable numérica original fue eliminada.

### 3.2 Feature Engineering

Se crearon cuatro variables para capturar patrones de comportamiento no lineales:

| Variable | Descripción |
|---|---|
| `riesgo_contrato` | `contract_type_mensual / (tenure_months + 1)` — vulnerabilidad de clientes sin compromiso contractual |
| `num_servicios` | Suma de columnas `has_*` — cantidad de servicios contratados |
| `cliente_problematico` | `(late_payments + tickets_grouped) > 0` — señal combinada de morosidad e incidencias |
| `anchor_score` | `tenure_months × num_products` — medida de compromiso del cliente |

### 3.3 Preprocesamiento

| Paso | Decisión |
|---|---|
| Encoding ordinal | `total_charges_cat` con orden explícito: Bajo → Medio-Bajo → Medio-Alto → Alto/VIP |
| Encoding nominal | One-Hot Encoding en `contract_type`, `payment_method`, `internet_service`, `region` con `drop_first=True` |
| Escalado | `StandardScaler` ajustado sobre train, aplicado sobre test |
| Split | 80% train / 20% test con `stratify=y` |
| Selección de features | `SelectFromModel` con `LogisticRegression(L1, C=0.1)`, reduciendo el espacio a las variables más informativas |

---

## 4. Modelos y Métricas

### 4.1 Baseline

Se compararon tres algoritmos con configuraciones por defecto, optimizando por F1-Score y ROC-AUC como métricas guía.

| Modelo | Resultado baseline |
|---|---|
| Logistic Regression | Lidera en F1-Score y ROC-AUC |
| Random Forest | No supera a Regresión Logística |
| XGBoost | Comparable, sin ventaja clara |

### 4.2 Optimización con RandomizedSearchCV

Se utilizó `StratifiedKFold` (5 folds) en todos los casos.

**Logistic Regression:** `penalty=L1`, `C=0.0335` → F1 en CV: **0.6090** (baseline: 0.566)

**Random Forest:** No superó a la Regresión Logística en las métricas relevantes. Descartado.

**XGBoost:** F1 similar a la Regresión Logística. Dado que esta última es más simple e interpretable, se priorizó.

### 4.3 Modelo Final

**Configuración:** `LogisticRegression(penalty='l1', solver='saga', C=0.5, class_weight={0:1, 1:1.6})`

#### Cross-Validation (5-fold, sobre train)

| Métrica | Media | Desvío |
|---|---|---|
| F1 | ~0.59 | bajo |
| ROC-AUC | ~0.76 | bajo |
| Precision | ~0.55 | bajo |
| Recall | ~0.64 | bajo |

#### Evaluación sobre Test Set

| Métrica | Valor |
|---|---|
| Precision (Churn) | 0.61 |
| Recall (Churn) | 0.70 |
| ROC-AUC | ~0.80 |

El modelo prioriza un recall moderadamente alto (detecta la mayoría de los churners) con una precisión razonable, lo que se traduce en campañas de retención más eficientes y menor costo operativo por intervención innecesaria.

---

## 5. Conclusiones

La Regresión Logística con regularización L1 y pesos de clase ajustados es la solución más adecuada para este problema. Su elección se justifica por:

- Mejor balance precisión-recall respecto a modelos más complejos.
- Interpretabilidad directa de coeficientes, facilitando la comunicación con el área de negocio.
- Estabilidad en validación cruzada (bajo desvío entre folds).

### Insights accionables

- Los clientes con **contrato mensual + pago en efectivo + internet móvil** concentran el mayor riesgo de churn.
- Implementar un **protocolo de fidelización proactiva a partir del tercer ticket de soporte** puede mitigar el salto crítico de abandono observado en el cuarto ticket.
- Los clientes **Alto/VIP** (mayor gasto acumulado) son el segmento más estable. El foco de retención debe estar en los segmentos Bajo y Medio-Bajo con contratos mensuales.
- La variable `region` no aporta valor predictivo y puede omitirse en versiones futuras del modelo.

---

## 6. Limitaciones

- **Dataset sintético:** el modelo fue entrenado con datos artificiales. Su desempeño sobre datos reales de producción podría diferir y requerirá validación adicional.
- **Desbalance de clases:** aunque mitigado con `class_weight`, el desbalance inherente limita el recall máximo alcanzable sin sacrificar precisión.
- **Linealidad del modelo:** la Regresión Logística asume relaciones lineales entre las variables y el log-odds del churn. Patrones no lineales complejos podrían no ser capturados completamente.
- **Ausencia de variables temporales:** el dataset no incluye series de tiempo del comportamiento del cliente, lo que impide capturar tendencias y estacionalidades.
- **Búsqueda de hiperparámetros no exhaustiva:** `RandomizedSearchCV` no garantiza el óptimo global. Una búsqueda bayesiana podría mejorar los resultados.
- **Umbral de decisión fijo en 0.5:** no se optimizó el threshold de clasificación. Ajustarlo según el costo relativo de falsos positivos vs. falsos negativos para el negocio podría mejorar el impacto operativo.
