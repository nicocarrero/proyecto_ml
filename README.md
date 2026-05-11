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

## Informe Técnico

El detalle completo del análisis, decisiones de modelado, feature engineering y limitaciones se encuentra en:

```text
reports/INFORME_TECNICO.md
```
