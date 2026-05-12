# proyecto_ml

Modelo de clasificación binaria para predecir la cancelación voluntaria de clientes (churn) en AndesLink Servicios Digitales S.A.

---

## Estructura del proyecto

```text
proyecto_ml/
├── README.md                 
├── data/
│   ├── raw/                  # Datos originales
│   └── processed/            # Datos procesados
├── models/                   # Modelos, métricas y visualizaciones
├── notebooks/
│   └── notebook.ipynb        # EDA y experimentación
├── reports/
│   └── INFORME_TECNICO.md
├── src/
│   ├── preprocessing.py      # Limpieza y transformación de datos
│   ├── feature_engineering.py
│   ├── train.py              # Entrenamiento del modelo
│   └── main.py               # Pipeline principal
└── environment.yml           # Dependencias del entorno
```

---

## Requisitos

* Conda

Todas las dependencias necesarias se encuentran definidas en:

```text
environment.yml
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://dagshub.com/carreronicoo/proyecto_ml.git <nombre_directorio>
cd <nombre_directorio>
git pull origin main
```

Reemplazar `<nombre_directorio>` por el nombre que desees para la carpeta local del proyecto.

---

### 2. Crear el entorno

```bash
conda env create -f environment.yml
conda activate proyecto_ml
```

---

### 3. Configurar acceso a DVC Remote

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user carreronicoo
dvc remote modify origin --local password 11c0bc53ab66e42295a8f9c55704bfec4c3580c0
```

---

### 4. Descargar datos y artefactos

```bash
dvc pull
```

Esto descarga:

* `data/raw/churn_sintetico.csv`
* `data/processed/data_final.csv`
* `models/model_pipeline.joblib`

---

## Ejecución del Pipeline

Para ejecutar todo el flujo completo de procesamiento y entrenamiento:

```bash
python src/main.py
```

El pipeline realiza automáticamente:

1. Preprocesamiento de datos
2. Feature Engineering
3. Entrenamiento del modelo
4. Evaluación
5. Generación de métricas y artefactos

---

## Modelo Final

El modelo seleccionado fue:

* Logistic Regression
* Regularización L1
* Ajuste de `class_weight`

---

## Resultados

| Métrica           | Valor  |
| ----------------- | ------ |
| Precision (Churn) | 0.61   |
| Recall (Churn)    | 0.70   |
| ROC-AUC           | 0.7981 |

Las métricas completas se encuentran en:

```text
models/metrics.json
```

La matriz de confusión se encuentra en:

```text
models/confusion_matrix.png
```

---

## Diccionario de Features (Dataset Final)

Para realizar inferencias con el modelo serializado (`models/model_pipeline.joblib`), el input debe respetar el siguiente esquema de variables generado luego del proceso de **Feature Engineering**.

| Tipo | Variables | Descripción, categorías y rangos |
|---|---|---|
| **Originales** | `tenure_months`, `monthly_charge`, `late_payments`, `avg_monthly_usage_gb`, `contract_type`, `payment_method`, `internet_service`, `has_streaming`, `has_security_pack`, `num_products`, `region`, `customer_age`, `is_promo` | **Variables numéricas:** <br> • `tenure_months`: 1 - 72 <br> • `monthly_charge`: 15.0 - 127.17 <br> • `late_payments`: 0 - 5 <br> • `avg_monthly_usage_gb`: 5.0 - 324.4 <br> • `customer_age`: 18 - 80 <br><br> **Variables categóricas:** <br> • `contract_type`: `mensual`, `anual`, `bianual` <br> • `payment_method`: `credito`, `debito`, `efectivo`, `transferencia` <br> • `internet_service`: `fibra`, `cable`, `movil`, `ninguno` <br> • `region`: `norte`, `sur`, `centro`, `oeste` <br><br> **Variables binarias:** <br> • `has_streaming`: `(0, 1)` <br> • `has_security_pack`: `(0, 1)` <br> • `is_promo`: `(0, 1)` |
| **Calculadas** | `total_charges_cat`, `tickets_grouped`, `riesgo_contrato`, `num_servicios`, `cliente_problematico`, `anchor_score` | Variables generadas durante el proceso de ingeniería de características. <br><br> • `total_charges_cat`: `Bajo`, `Medio-Bajo`, `Medio-Alto`, `Alto/VIP` <br> • `tickets_grouped`: 0 - 5 <br> • `riesgo_contrato`: 0.0 - 1.0 <br> • `num_servicios`: 0 - 2 <br> • `cliente_problematico`: `(0, 1)` <br> • `anchor_score`: 1 - 288 |

> **Nota:**  
> El modelo espera como entrada el dataset final luego del proceso de Feature Engineering.  
> Algunas variables calculadas derivan de columnas originales que ya no existen en el dataset final, por lo que esta sección se incluye únicamente como referencia para pruebas e inferencias manuales.

---

## Informe Técnico

El detalle completo del análisis, decisiones de modelado, feature engineering y limitaciones se encuentra en:

```text
reports/INFORME_TECNICO.md
```

---

## Notebook de Exploración

Si deseas revisar el análisis exploratorio de datos (EDA), feature engineering, pruebas de modelos y experimentación realizada durante el desarrollo, puedes abrir el notebook:

```text
notebooks/notebook.ipynb
```

---

