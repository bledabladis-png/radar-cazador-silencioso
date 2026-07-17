# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy.stats import kruskal
from src.utils import get_col
from indicators.mte import score_scenarios, sector_rotation_score, safe_haven_score, credit_stress_score, inflation_pressure_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VALIDACION DESCRIPTIVA EX POST - MTE v1.0 (HISTORICO COMPLETO)")
print("=" * 70)

df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
spy_close = get_col(df_market, 'SPY', 'Close')

start_date = df_market.index[0]
end_date = df_market.index[-1]
eval_dates = pd.date_range(start_date, end_date, freq='W-FRI')
eval_dates = [d for d in eval_dates if d in df_market.index]

print(f"Evaluando {len(eval_dates)} semanas ({start_date.date()} a {end_date.date()})...")

rows = []
for i, date in enumerate(eval_dates):
    if i % 50 == 0:
        print(f"  Progreso: {i}/{len(eval_dates)}")
    idx = df_market.index.get_loc(date)
    df_slice = df_market.iloc[:idx+1]
    
    try:
        srs = sector_rotation_score(df_slice)
        shs = safe_haven_score(df_slice)
        try:
            vix_close = get_col(df_slice, '^VIX', 'Close')
            vix_ret = vix_close.pct_change().rolling(20).std().iloc[-1]
            vix_ma = vix_close.pct_change().rolling(60).std().mean()
            fc_approx = float(np.clip(np.tanh((vix_ret / (vix_ma + 1e-9) - 1) / 2), 0, 1))
            hyg = get_col(df_slice, 'HYG', 'Close')
            lqd = get_col(df_slice, 'LQD', 'Close')
            spread = hyg / lqd
            cred_approx = float(np.clip(np.tanh(-(spread.pct_change(20).iloc[-1]) / 2), 0, 1))
            cls = float(np.mean([fc_approx, cred_approx, 0.3, 0.3]))
        except:
            cls = 0.3
        ips = inflation_pressure_score(df_slice)
        
        scores = score_scenarios(srs, shs, cls, ips)
        scenario = max(scores, key=scores.get)
        
        spy_idx = spy_close.index.get_loc(date)
        ret_3m = spy_close.iloc[min(spy_idx+63, len(spy_close)-1)] / spy_close.iloc[spy_idx] - 1 if spy_idx+63 < len(spy_close) else np.nan
        ret_6m = spy_close.iloc[min(spy_idx+126, len(spy_close)-1)] / spy_close.iloc[spy_idx] - 1 if spy_idx+126 < len(spy_close) else np.nan
        
        rows.append({'date': date, 'scenario': scenario, 'ret_3m': ret_3m, 'ret_6m': ret_6m})
    except:
        pass

df = pd.DataFrame(rows)
print(f"\nRegistros: {len(df)}")

print("\n" + "="*70)
print("RETORNOS DEL SPY POR ESCENARIO (con dispersion)")
print("="*70)

scenarios_with_data = []
for scenario in ['EXPANSION', 'SOFT LANDING', 'RECESSION', 'STAGFLATION', 'CRISIS', 'MIXED']:
    subset = df[df['scenario'] == scenario]
    if len(subset) >= 3:
        scenarios_with_data.append(scenario)
        print(f"\n{scenario} (n={len(subset)}):")
        for col in ['ret_3m', 'ret_6m']:
            vals = subset[col].dropna()
            if len(vals) > 0:
                print(f"  {col}: media={vals.mean()*100:+.2f}%  mediana={vals.median()*100:+.2f}%  std={vals.std()*100:.2f}%  min={vals.min()*100:+.2f}%  max={vals.max()*100:+.2f}%")
    elif len(subset) > 0:
        print(f"\n{scenario} (n={len(subset)}): muestra insuficiente")
    else:
        print(f"\n{scenario}: sin observaciones")

print("\n" + "="*70)
print("TEST DE KRUSKAL-WALLIS")
print("="*70)

if len(scenarios_with_data) >= 2:
    for col in ['ret_3m', 'ret_6m']:
        groups = [df[df['scenario']==s][col].dropna().values for s in scenarios_with_data if len(df[df['scenario']==s]) >= 3]
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
            print(f"  {col}: H={stat:.2f}, p={p:.4f}  {'✓ Significativo' if p < 0.05 else '⚠️ No significativo'}")

print("\n" + "="*70)
print("BOOTSTRAP IC 95% (retorno a 3 meses)")
print("="*70)

for scenario in scenarios_with_data:
    vals = df[df['scenario']==scenario]['ret_3m'].dropna().values
    if len(vals) >= 5:
        boot_means = []
        for _ in range(5000):
            sample = np.random.choice(vals, size=len(vals), replace=True)
            boot_means.append(sample.mean())
        boot_means = np.array(boot_means)
        low, high = np.percentile(boot_means, [2.5, 97.5])
        print(f"  {scenario:<15} media={vals.mean()*100:+.2f}%  IC95=[{low*100:+.2f}%, {high*100:+.2f}%]")

print("\n" + "="*70)
print(f"Periodo: {start_date.date()} a {end_date.date()} ({len(eval_dates)} semanas)")
print("CLS aproximado con VIX y HYG/LQD. Validacion descriptiva, no predictiva.")
print("="*70)
