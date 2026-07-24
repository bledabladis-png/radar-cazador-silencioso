# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from src.utils import get_col
from indicators.mte import sector_rotation_score, safe_haven_score, credit_stress_score, inflation_pressure_score, score_scenarios
from indicators.breadth_equity import compute_advance_decline
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("AUDITORÍA DE INDEPENDENCIA: SLPM ↔ MTE ↔ SRS")
print("=" * 70)

# Cargar datos
df_slpm = pd.read_csv('outputs/slpm_history_v11.csv', parse_dates=['date'])
df_slpm = df_slpm[df_slpm['n_leaders'] > 0].copy()
print(f"  SLPM: {len(df_slpm)} observaciones")

df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
nfci = pd.read_csv('data/macro_manual/nfci.csv', index_col=0, parse_dates=True)['NFCI']
oas = pd.read_csv('data/macro_manual/credit_oas.csv', index_col=0, parse_dates=True)['CreditOAS']
spy_close = get_col(df_market, 'SPY', 'Close')

# Calcular MTE y SRS para cada fecha del SLPM
print("  Calculando MTE y SRS para cada fecha...")
mte_scores = []
for i, date in enumerate(df_slpm['date']):
    if i % 50 == 0:
        print(f"    Progreso: {i}/{len(df_slpm)}")
    if date not in df_market.index:
        continue
    idx = df_market.index.get_loc(date)
    df_slice = df_market.iloc[:idx+1]
    try:
        srs = sector_rotation_score(df_slice)
        shs = safe_haven_score(df_slice)
        ips = inflation_pressure_score(df_slice)
        
        vix_close = get_col(df_slice, '^VIX', 'Close')
        vix_ret = vix_close.pct_change().rolling(20).std().iloc[-1]
        vix_ma = vix_close.pct_change().rolling(60).std().mean()
        fc = -float(np.clip(np.tanh((vix_ret / (vix_ma + 1e-9) - 1) / 2), 0, 1))
        
        hyg = get_col(df_slice, 'HYG', 'Close')
        lqd = get_col(df_slice, 'LQD', 'Close')
        credit_signal = float(np.clip(np.tanh((hyg.iloc[-1] / lqd.iloc[-1] - 1) / 2), -1, 1))
        
        nfci_series = nfci.loc[:date] if date in nfci.index else None
        oas_series = oas.loc[:date] if date in oas.index else None
        cls = credit_stress_score(fc, credit_signal, vix_ret, vix_ma, 0.5, 0.5, nfci_series, oas_series)
        
        scores = score_scenarios(srs, shs, cls, ips)
        mte_scenario = max(scores, key=scores.get)
        
        mte_scores.append({
            'date': date,
            'srs': srs,
            'shs': shs,
            'cls': cls,
            'ips': ips,
            'mte_scenario': mte_scenario
        })
    except Exception as e:
        pass

df_mte = pd.DataFrame(mte_scores)
merged = df_slpm.merge(df_mte, on='date', how='inner')
print(f"\n  Observaciones comunes: {len(merged)}")

# ============================================================
# 1. CORRELACIONES
# ============================================================
print("\n" + "="*70)
print("1. CORRELACIONES (Spearman)")
print("="*70)

slpm_cols = ['leader_breadth', 'flow_divergence', 'structural_score']
mte_cols = ['srs', 'shs', 'cls', 'ips']

for slpm_col in slpm_cols:
    for mte_col in mte_cols:
        valid = merged[[slpm_col, mte_col]].dropna()
        if len(valid) > 10:
            rho, p = spearmanr(valid[slpm_col], valid[mte_col])
            if abs(rho) > 0.80:
                flag = ' ⚠️ ALTA CORRELACIÓN'
            elif abs(rho) > 0.50:
                flag = ' (moderada)'
            else:
                flag = ' ✓ Independiente'
            print(f"  {slpm_col:<25} ↔ {mte_col:<5} ρ={rho:+.3f} (p={p:.4f}){flag}")

# ============================================================
# 2. TABLA DE CONTINGENCIA: SLPM vs MTE
# ============================================================
print("\n" + "="*70)
print("2. TABLA DE CONTINGENCIA: SLPM vs MTE")
print("="*70)

ct = pd.crosstab(merged['state'], merged['mte_scenario'], normalize='index') * 100
print(ct.round(1).to_string())

# ============================================================
# 3. ¿EL SLPM AÑADE INFORMACIÓN INDEPENDIENTE?
# ============================================================
print("\n" + "="*70)
print("3. ¿EL SLPM AÑADE INFORMACIÓN INDEPENDIENTE?")
print("="*70)

# Si el SLPM solo replica el MTE, todos los escenarios MTE deberían tener la misma distribución SLPM
# Si hay variación, el SLPM añade información

mte_scenarios = merged['mte_scenario'].unique()
for scenario in mte_scenarios:
    subset = merged[merged['mte_scenario'] == scenario]
    if len(subset) >= 5:
        dist = subset['state'].value_counts(normalize=True) * 100
        print(f"\n  MTE = {scenario} (n={len(subset)}):")
        for state in ['LEADERSHIP_CONFIRMED', 'TACTICAL_CORRECTION', 'STRUCTURAL_DETERIORATION']:
            pct = dist.get(state, 0)
            bar = '█' * int(pct / 2)
            print(f"    {state:<28} {pct:5.1f}%  {bar}")

print("\n" + "="*70)
print("AUDITORÍA DE INDEPENDENCIA COMPLETADA")
print("="*70)
