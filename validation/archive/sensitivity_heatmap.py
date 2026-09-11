import pandas as pd

# Cargar resultados del estudio de sensibilidad (si existe)
try:
    df = pd.read_csv('outputs/audit/sensitivity_results.csv')
    print("=== SENSIBILIDAD DE VENTANAS Y PESOS ===")
    print(df.to_string())
except FileNotFoundError:
    print("No se encontró sensitivity_results.csv. Ejecute primero el estudio de sensibilidad.")
