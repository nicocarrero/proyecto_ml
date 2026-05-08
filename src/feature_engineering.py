# src/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import os

def run_feature_engineering(df):
    """
    Feature Engineering para el dataset de churn. Se crean nuevas variables basadas en las originales, se agrupan categorías y se eliminan columnas redundantes.
    """
    df = df.copy()
    
    #---Trasnformaciones de variables---

    # 1. Segmentación de Total Charges en categorías usando cuartiles
    q1 = df['total_charges'].quantile(0.25)
    q2 = df['total_charges'].quantile(0.50)
    q3 = df['total_charges'].quantile(0.75)

    def segmentar_total_charges(x):
        if x <= q1: return 'Bajo'
        if x <= q2: return 'Medio-Bajo'
        if x <= q3: return 'Medio-Alto'
        return 'Alto/VIP'

    df['total_charges_cat'] = df['total_charges'].apply(segmentar_total_charges)

    
    # 2. Agrupación de Tickets de Soporte
    df['tickets_grouped'] = df['support_tickets'].copy()
    df.loc[df['tickets_grouped'] >= 5, 'tickets_grouped'] = 5
    
# ----
    #---Nuevas variables---

    # 3. Riesgo de Contrato
    if 'contract_type' in df.columns and 'tenure_months' in df.columns:
        es_mensual = (df['contract_type'] == 'mensual').astype(int)
        df['riesgo_contrato'] = es_mensual / (df['tenure_months'] + 1)

    # 4. Conteo de Servicios
    cols_servicios = [col for col in df.columns if col.startswith('has_')]
    if cols_servicios:
        df['num_servicios'] = df[cols_servicios].sum(axis=1)
    else:
        df['num_servicios'] = 0

    # 5. Cliente Problemático y Anchor Score
    if 'late_payments' in df.columns:
        df['cliente_problematico'] = ((df['late_payments'] + df['tickets_grouped']) > 0).astype(int)
    
    if 'num_products' in df.columns and 'tenure_months' in df.columns:
        df['anchor_score'] = df['tenure_months'] * df['num_products']
        
# ----

    # 6. Drop de de columnas originales modificadas 
    cols_to_drop = ['total_charges', 'support_tickets']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return df

# ----


if __name__ == "__main__":
    # Rutas relativas para reproducibilidad en Git
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "raw", "churn_sintetico.csv")
    output_path = os.path.join(base_dir, "data", "processed", "data_final.csv")
    
    try:
        data = pd.read_csv(input_path)
        df_final = run_feature_engineering(data)
        df_final.to_csv(output_path, index=False)
        print(f"✓ Feature Engineering completado. Columnas finales: {list(df_final.columns)}")
        print(f"--> Shape: {df_final.shape}")
    except Exception as e:
        print(f"❌ Error al procesar: {e}")