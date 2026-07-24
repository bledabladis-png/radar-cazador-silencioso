# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VALIDACIÓN ESTADÍSTICA DEL SLPM v1.1")
print("=" * 70)

# Cargar histórico v1.1
df = pd.read_csv('outputs/slpm_history_v11.csv', parse_dates=['date'])
print(f"  Registros totales: {len(df)}")

# Filtrar solo semanas con líderes
df_valid = df[df['n_leaders'] > 0].copy()
print(f"  Registros con líderes (válidos): {len(df_valid)}")

cols = ['leader_breadth', 'flow_divergence', 'structural_score']

# ============================================================
# 0. COBERTURA
# ============================================================
print("\n" + "="*70 + "\n0. COBERTURA\n" + "="*70)
for col in cols:
    nan_pct = df_valid[col].isna().mean() * 100
    inf_pct = np.isinf(df_valid[col]).mean() * 100 if df_valid[col].dtype in [np.float64, np.float32] else 0
    print(f"  {col:<25} NaN={nan_pct:5.2f}%  Inf={inf_pct:5.2f}%")

# ============================================================
# 1. ESTACIONARIEDAD (ADF)
# ============================================================
print("\n" + "="*70 + "\n1. ESTACIONARIEDAD (ADF)\n" + "="*70)
for col in cols:
    try:
        stat, p, *_ = adfuller(df_valid[col].dropna())
        status = '✓ Estacionaria' if p < 0.05 else '⚠️ No estacionaria'
        print(f"  {col:<25} p={p:.4f}  {status}")
    except ValueError:
        print(f"  {col:<25} serie constante")

# ============================================================
# 2. AUTOCORRELACIÓN Y N_eff
# ============================================================
print("\n" + "="*70 + "\n2. AUTOCORRELACIÓN Y N_eff\n" + "="*70)
for col in cols:
    if len(df_valid) > 10:
        ac = df_valid[col].autocorr()
        if pd.notna(ac):
            N = len(df_valid[col].dropna())
            Neff_raw = N * (1 - ac) / (1 + ac) if ac != -1 else N
            Neff = min(Neff_raw, N)
            status = '✓ Reactivo' if ac < 0.70 else '✓ Alta (esperable)' if ac < 0.90 else '⚠️ Muy alta'
            print(f"  {col:<25} autocorr={ac:.3f}  N={N}  N_eff={Neff:.0f}  {status}")

# ============================================================
# 3. BOOTSTRAP DEL STRUCTURAL SCORE
# ============================================================
print("\n" + "="*70 + "\n3. BOOTSTRAP DEL STRUCTURAL SCORE (500 remuestreos)\n" + "="*70)
col = 'structural_score'
if len(df_valid) > 10:
    means = []
    for _ in range(500):
        sample = df_valid[col].sample(frac=1, replace=True)
        means.append(sample.mean())
    means = np.array(means)
    bias = means.mean() - df_valid[col].mean()
    print(f"  media={df_valid[col].mean():.3f}  boot_mean={means.mean():.3f}  sesgo={bias:.4f}  IC95=[{np.percentile(means,2.5):.3f}, {np.percentile(means,97.5):.3f}]  {'✓' if abs(bias)<0.05 else '⚠️'}")

# ============================================================
# 4. ESTABILIDAD DEL CLASIFICADOR
# ============================================================
print("\n" + "="*70 + "\n4. ESTABILIDAD DEL CLASIFICADOR\n" + "="*70)
state_counts = df_valid['state'].value_counts()
print("  Distribución de estados:")
for state, count in state_counts.items():
    pct = count / len(df_valid) * 100
    bar = '█' * int(pct / 2)
    print(f"    {state:<28} {count:>4} ({pct:5.1f}%)  {bar}")

# Duración media de cada estado
print("\n  Duración media de rachas (semanas):")
for state in state_counts.index:
    mask = df_valid['state'] == state
    rachas = (mask != mask.shift()).cumsum()
    durations = rachas[mask].value_counts()
    print(f"    {state:<28} media={durations.mean():.1f}  mediana={durations.median():.0f}  máx={durations.max():.0f}")

# Matriz de transición
print("\n  Matriz de transición (probabilidad %):")
states = list(state_counts.index)
tm = pd.DataFrame(0, index=states, columns=states, dtype=float)
for a, b in zip(df_valid['state'][:-1], df_valid['state'][1:]):
    tm.loc[a, b] += 1
tm = tm.div(tm.sum(axis=1), axis=0) * 100
print(tm.round(1).to_string())

# ============================================================
# 5. COHERENCIA INTERNA DEL CLASIFICADOR
# ============================================================
print("\n" + "="*70 + "\n5. COHERENCIA INTERNA\n" + "="*70)
for state in states:
    subset = df_valid[df_valid['state'] == state]
    print(f"\n  {state} (n={len(subset)}):")
    print(f"    leader_breadth medio:   {subset['leader_breadth'].mean():.3f}")
    print(f"    flow_divergence medio:  {subset['flow_divergence'].mean():+.3f}")
    print(f"    structural_score medio: {subset['structural_score'].mean():.3f}")

# Verificar que LEADERSHIP_CONFIRMED tiene los valores más altos
print("\n  Orden esperado: LEADERSHIP_CONFIRMED > TACTICAL_CORRECTION > STRUCTURAL_DETERIORATION")
for col in cols:
    means = df_valid.groupby('state')[col].mean()
    if means.get('LEADERSHIP_CONFIRMED', 0) > means.get('STRUCTURAL_DETERIORATION', 0):
        print(f"  ✓ {col}: orden correcto")
    else:
        print(f"  ✗ {col}: orden invertido")

# Comparación con v1.0
try:
    df_v10 = pd.read_csv('outputs/slpm_history.csv', parse_dates=['date'])
    df_v10 = df_v10[df_v10['n_leaders'] > 0]
    common = df_v10.merge(df_valid[['date', 'state']], on='date', suffixes=('_v10', '_v11'))
    changes = (common['state_v10'] != common['state_v11']).sum()
    print(f"\nComparación v1.0 vs v1.1 ({len(common)} semanas comunes):")
    print(f"  Cambios de estado: {changes}/{len(common)} ({changes/len(common)*100:.1f}%)")
except:
    pass

print("\n" + "="*70)
print("VALIDACIÓN COMPLETADA")
print("="*70)
