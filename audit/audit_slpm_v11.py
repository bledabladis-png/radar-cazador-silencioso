# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("AUDITORÍA COMPLETA DEL SLPM v1.1")
print("=" * 70)

# Cargar datos
df_v10 = pd.read_csv('outputs/slpm_history.csv', parse_dates=['date'])
df_v11 = pd.read_csv('outputs/slpm_history_v11.csv', parse_dates=['date'])
df_v10 = df_v10[df_v10['n_leaders'] > 0]
df_v11 = df_v11[df_v11['n_leaders'] > 0]

# ============================================================
# PRUEBA 1: MATRIZ DE TRANSICIÓN v1.0 → v1.1
# ============================================================
print("\n" + "=" * 70)
print("PRUEBA 1: MATRIZ DE TRANSICIÓN v1.0 → v1.1")
print("=" * 70)

merged = df_v10.merge(df_v11[['date', 'state']], on='date', suffixes=('_v10', '_v11'))
changes = merged[merged['state_v10'] != merged['state_v11']]

print(f"\nTotal cambios: {len(changes)}/{len(merged)} ({len(changes)/len(merged)*100:.1f}%)")
print("\nDirección de los cambios:")
ct = pd.crosstab(merged['state_v10'], merged['state_v11'], margins=True)
print(ct.to_string())

# ============================================================
# PRUEBA 2: SENSIBILIDAD DE PESOS Y UMBRALES
# ============================================================
print("\n" + "=" * 70)
print("PRUEBA 2: SENSIBILIDAD DE PESOS Y UMBRALES")
print("=" * 70)

def classify_with_params(lb, lfd, w_lb, threshold):
    score = w_lb * (lb - 0.5) * 2 + (1 - w_lb) * np.tanh(lfd * 2)
    if score > threshold and lb >= 0.4:
        return 'LEADERSHIP_CONFIRMED'
    elif score < -threshold and lb < 0.4:
        return 'STRUCTURAL_DETERIORATION'
    return 'TACTICAL_CORRECTION'

variations = []
for w_lb in [0.50, 0.55, 0.60, 0.65, 0.70]:
    for threshold in [0.15, 0.20, 0.25]:
        df_valid = df_v11.copy()
        df_valid['test_state'] = df_valid.apply(
            lambda row: classify_with_params(row['leader_breadth'], row['flow_divergence'], w_lb, threshold), axis=1
        )
        changes_vs_base = (df_valid['test_state'] != df_valid['state']).sum()
        variations.append({'w_lb': w_lb, 'threshold': threshold, 'changes': changes_vs_base, 'pct': changes_vs_base/len(df_valid)*100})

df_var = pd.DataFrame(variations)
print("\nCambios respecto al modelo base (0.60, 0.20):")
print(df_var.to_string(index=False))

most_stable = df_var.loc[df_var['changes'].idxmin()]
print(f"\nCombinacion mas estable: w_lb={most_stable['w_lb']}, threshold={most_stable['threshold']} ({most_stable['changes']} cambios, {most_stable['pct']:.1f}%)")

# ============================================================
# PRUEBA 3: ESTABILIDAD TEMPORAL POR SUBPERÍODOS
# ============================================================
print("\n" + "=" * 70)
print("PRUEBA 3: ESTABILIDAD TEMPORAL POR SUBPERIODOS")
print("=" * 70)

df_valid = df_v11.copy()
df_valid['year'] = pd.to_datetime(df_valid['date']).dt.year

periods = {
    '2021-2022': [2021, 2022],
    '2023-2024': [2023, 2024],
    '2025-2026': [2025, 2026]
}

for period_name, years in periods.items():
    subset = df_valid[df_valid['year'].isin(years)]
    if len(subset) > 0:
        dist = subset['state'].value_counts(normalize=True).sort_index()
        print(f"\n{period_name} (n={len(subset)}):")
        for state, pct in dist.items():
            bar = '█' * int(pct * 50)
            print(f"  {state:<28} {pct*100:5.1f}%  {bar}")

# ============================================================
# PRUEBA 4: AUDITORÍA DE LA RACHA DE 54 SEMANAS
# ============================================================
print("\n" + "=" * 70)
print("PRUEBA 4: AUDITORIA DE LA RACHA DE 54 SEMANAS EN STRUCTURAL_DETERIORATION")
print("=" * 70)

mask = df_valid['state'] == 'STRUCTURAL_DETERIORATION'
rachas = (mask != mask.shift()).cumsum()
durations = rachas[mask].value_counts().sort_values(ascending=False)

longest_racha_id = durations.index[0]
longest_racha = df_valid[rachas == longest_racha_id]

print(f"\nRacha mas larga: {len(longest_racha)} semanas")
print(f"Periodo: {longest_racha['date'].iloc[0].date()} -> {longest_racha['date'].iloc[-1].date()}")
print(f"Sector lider predominante: {longest_racha['sector_etf'].mode()[0]}")
print(f"\nLeader Breadth medio: {longest_racha['leader_breadth'].mean():.3f}")
print(f"Flow Divergence medio: {longest_racha['flow_divergence'].mean():+.3f}")

sector_changes = (longest_racha['sector_etf'] != longest_racha['sector_etf'].shift()).sum()
print(f"\nCambios de sector lider durante la racha: {sector_changes}")
print("Interpretacion: ", end="")
if sector_changes == 0:
    print("Un unico sector permanecio lider mientras sus lideres internos se deterioraban (escenario esperado para SLPM).")
else:
    print("Varios sectores se turnaron en el liderazgo, todos con deterioro interno (posible regimen de mercado adverso generalizado).")

print("\n" + "=" * 70)
print("AUDITORIA COMPLETADA")
print("=" * 70)
