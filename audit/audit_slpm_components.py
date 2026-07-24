# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("AUDITORÍA DE CONTRIBUCIÓN DE COMPONENTES — SLPM")
print("=" * 70)

# Cargar histórico válido
df = pd.read_csv('outputs/slpm_history.csv', parse_dates=['date'])
df_valid = df[df['n_leaders'] > 0].copy()
print(f"  Observaciones: {len(df_valid)}")

# Normalizar componentes (igual que en structural_leadership.py)
df_valid['struct_rs_norm'] = np.tanh(df_valid['struct_rs'] * 5)
df_valid['leader_breadth_norm'] = (df_valid['leader_breadth'] - 0.5) * 2
df_valid['lfd_norm'] = np.tanh(df_valid['flow_divergence'] * 2)

# Modelos de prueba
modelos = {
    'A: Solo Leader Breadth':           [0.0, 1.0, 0.0],
    'B: Solo Flow Divergence':          [0.0, 0.0, 1.0],
    'C: Leader Breadth + Flow Div.':    [0.0, 0.6, 0.4],
    'D: struct_rs + Leader Breadth':    [0.5, 0.5, 0.0],
    'E: struct_rs + Flow Div.':         [0.5, 0.0, 0.5],
    'F: MODELO COMPLETO (actual)':      [0.3, 0.4, 0.3],
}

print("\n" + "="*70)
print("CAPACIDAD DISCRIMINATIVA DE CADA MODELO")
print("(Diferencia entre LEADERSHIP_CONFIRMED y STRUCTURAL_DETERIORATION)")
print("="*70)

for nombre, (w_rs, w_lb, w_lfd) in modelos.items():
    df_valid['score_test'] = (
        w_rs * df_valid['struct_rs_norm'] +
        w_lb * df_valid['leader_breadth_norm'] +
        w_lfd * df_valid['lfd_norm']
    )
    
    confirmed = df_valid[df_valid['state'] == 'LEADERSHIP_CONFIRMED']['score_test'].mean()
    deteriorated = df_valid[df_valid['state'] == 'STRUCTURAL_DETERIORATION']['score_test'].mean()
    diff = confirmed - deteriorated
    
    bar = '█' * int(max(0, diff * 50))
    print(f"  {nombre:<35} diff={diff:+.3f}  {bar}")

print("\n" + "="*70)
print("CONCLUSIÓN")
print("="*70)
print("  El modelo con mayor diferencia entre estados es el que mejor discrimina.")
print("  Si struct_rs reduce la diferencia respecto a un modelo sin él, es redundante.")
print("="*70)
