import pandas as pd
import numpy as np

df = pd.read_csv('outputs/audit/backtest_v2_results.csv', parse_dates=['date'])
df = df.sort_values('date')
df['next_obtained'] = df['obtained'].shift(-1)
# Eliminar última fila (sin siguiente)
df = df.dropna(subset=['next_obtained'])

# Matriz de transición
trans = pd.crosstab(df['obtained'], df['next_obtained'], normalize='index')
print("=== MATRIZ DE TRANSICIÓN DE REGÍMENES (probabilidad de pasar de fila a columna) ===")
print(trans.to_string(float_format=lambda x: f'{x:.1%}'))
