# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.utils import get_col
from indicators.mte import (
    sector_rotation_score, safe_haven_score, credit_stress_score,
    inflation_pressure_score, score_scenarios, compute_msi, compute_ipi
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VALIDACIÓN COMPLETA MTE v1.0 vs v1.1 (520 semanas)")
print("=" * 70)

# Cargar datos
df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
nfci = pd.read_csv('data/macro_manual/nfci.csv', index_col=0, parse_dates=True)['NFCI']
oas = pd.read_csv('data/macro_manual/credit_oas.csv', index_col=0, parse_dates=True)['CreditOAS']

start_date = df_market.index[0]
end_date = df_market.index[-1]
eval_dates = pd.date_range(start_date, end_date, freq='W-FRI')
eval_dates = [d for d in eval_dates if d in df_market.index]

print(f"Evaluando {len(eval_dates)} semanas...")

rows = []
for i, date in enumerate(eval_dates):
    if i % 50 == 0:
        print(f"  Progreso: {i}/{len(eval_dates)}")
    idx = df_market.index.get_loc(date)
    df_slice = df_market.iloc[:idx+1]
    
    try:
        srs = sector_rotation_score(df_slice)
        shs = safe_haven_score(df_slice)
        ips = inflation_pressure_score(df_slice)
        
        # CLS v1.0 (sintético)
        vix_close = get_col(df_slice, '^VIX', 'Close')
        vix_ret = vix_close.pct_change().rolling(20).std().iloc[-1]
        vix_ma = vix_close.pct_change().rolling(60).std().mean()
        fc_approx = float(np.clip(np.tanh((vix_ret / (vix_ma + 1e-9) - 1) / 2), 0, 1))
        hyg = get_col(df_slice, 'HYG', 'Close')
        lqd = get_col(df_slice, 'LQD', 'Close')
        spread = hyg / lqd
        cred_approx = float(np.clip(np.tanh(-(spread.pct_change(20).iloc[-1]) / 2), 0, 1))
        cls_v10 = float(np.sqrt(np.mean(np.square([fc_approx, cred_approx, 0.3, 0.3]))))
        
        # CLS v1.1 (con FRED)
        nfci_window = nfci.loc[:date] if date in nfci.index else nfci
        oas_window = oas.loc[:date] if date in oas.index else oas
        
        def robust_zscore_series(series, window=104):
            median = series.rolling(window, min_periods=20).median()
            def mad_func(x):
                return np.median(np.abs(x - np.median(x)))
            mad = series.rolling(window, min_periods=20).apply(mad_func, raw=True)
            return (series - median) / (1.4826 * mad + 1e-9)
        
        def stress_transform(z):
            return float(np.clip(np.tanh(z.iloc[-1] / 2.0), 0, 1)) if len(z) > 0 else 0.5
        
        nfci_z = robust_zscore_series(nfci_window)
        nfci_stress = stress_transform(nfci_z)
        oas_z = robust_zscore_series(oas_window)
        oas_stress = stress_transform(oas_z)
        
        credit_family = 0.60 * oas_stress + 0.40 * fc_approx
        vix_stress = float(np.clip(np.tanh(vix_ret / 2), 0, 1))
        
        cls_v11 = (0.25 * nfci_stress + 0.35 * credit_family + 0.25 * vix_stress + 0.15 * 0.5)
        
        # Escenarios
        scores_v10 = score_scenarios(srs, shs, cls_v10, ips)
        scenario_v10 = max(scores_v10, key=scores_v10.get)
        
        scores_v11 = score_scenarios(srs, shs, cls_v11, ips)
        scenario_v11 = max(scores_v11, key=scores_v11.get)
        
        rows.append({
            'date': date,
            'srs': srs, 'shs': shs, 'ips': ips,
            'cls_v10': cls_v10, 'cls_v11': cls_v11,
            'scenario_v10': scenario_v10, 'scenario_v11': scenario_v11
        })
    except Exception as e:
        if i < 5:
            print(f"  Error en {date.date()}: {e}")
        continue

df = pd.DataFrame(rows)
print(f"\nRegistros válidos: {len(df)}")

# ── DISTRIBUCIÓN DE ESCENARIOS ──
print("\n" + "="*70)
print("DISTRIBUCIÓN DE ESCENARIOS")
print("="*70)
for version, col in [('v1.0', 'scenario_v10'), ('v1.1', 'scenario_v11')]:
    dist = df[col].value_counts(normalize=True).sort_index()
    print(f"\n  {version}:")
    for s, pct in dist.items():
        bar = '█' * int(pct * 50)
        print(f"    {s:<15} {pct*100:5.1f}%  {bar}")

# ── MATRIZ DE CONFUSIÓN ──
print("\n" + "="*70)
print("MATRIZ DE CONFUSIÓN v1.0 → v1.1 (%)")
print("="*70)
ct = pd.crosstab(df['scenario_v10'], df['scenario_v11'], normalize='index') * 100
print(ct.round(1).to_string())

# ── CAMBIOS DE ESCENARIO ──
changed = df['scenario_v10'] != df['scenario_v11']
print(f"\nCambios de escenario: {changed.sum()}/{len(df)} ({changed.mean()*100:.1f}%)")

# ── DETECCIÓN DE ESTRÉS (CLS > 0.50) ──
print("\n" + "="*70)
print("SEMANAS CON CLS v1.1 > 0.50 (umbral CRISIS)")
print("="*70)
high_cls = df[df['cls_v11'] > 0.50]
if len(high_cls) > 0:
    print(f"  Total: {len(high_cls)} semanas")
    print(f"\n  Primeras 10 fechas:")
    for _, row in high_cls.head(10).iterrows():
        print(f"    {row['date'].date()}  CLS_v10={row['cls_v10']:.3f}  CLS_v11={row['cls_v11']:.3f}  v1.0={row['scenario_v10']}  v1.1={row['scenario_v11']}")
else:
    print("  Ninguna semana supera 0.50")

# ── FECHAS CLAVE ──
print("\n" + "="*70)
print("FECHAS CLAVE CON CLS v1.1")
print("="*70)
for date_str in ['2020-03-16', '2020-03-23', '2022-06-15', '2023-07-31']:
    row = df[df['date'] == date_str]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"\n  {date_str}:")
        print(f"    CLS v1.0={r['cls_v10']:.3f} → {r['scenario_v10']}")
        print(f"    CLS v1.1={r['cls_v11']:.3f} → {r['scenario_v11']}")

print("\n" + "="*70)
print("VALIDACIÓN COMPLETADA")
print("="*70)
