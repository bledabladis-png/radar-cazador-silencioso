# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.utils import get_col
from indicators.mte import (
    sector_rotation_score, safe_haven_score, credit_stress_score,
    inflation_pressure_score, score_scenarios
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DIAGNÓSTICO MTE v1.1 EN FECHAS CLAVE")
print("=" * 70)

df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
nfci = pd.read_csv('data/macro_manual/nfci.csv', index_col=0, parse_dates=True)['NFCI']
oas = pd.read_csv('data/macro_manual/credit_oas.csv', index_col=0, parse_dates=True)['CreditOAS']

test_dates = ['2020-03-16', '2020-03-23', '2022-06-15', '2023-07-31', '2024-01-02', '2026-07-16']

for date_str in test_dates:
    if date_str not in df_market.index:
        print(f"\n{date_str}: fecha no disponible")
        continue
    
    idx = df_market.index.get_loc(date_str)
    df_slice = df_market.iloc[:idx+1]
    
    try:
        srs = sector_rotation_score(df_slice)
        shs = safe_haven_score(df_slice)
        ips = inflation_pressure_score(df_slice)
        
        vix_close = get_col(df_slice, '^VIX', 'Close')
        vix_ret = vix_close.pct_change().rolling(20).std().iloc[-1]
        vix_ma = vix_close.pct_change().rolling(60).std().mean()
        fc_approx = float(np.clip(np.tanh((vix_ret / (vix_ma + 1e-9) - 1) / 2), 0, 1))
        
        nfci_window = nfci.loc[:date_str]
        oas_window = oas.loc[:date_str]
        
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
        cls = (0.25 * nfci_stress + 0.35 * credit_family + 0.25 * vix_stress + 0.15 * 0.5)
        
        scores = score_scenarios(srs, shs, cls, ips)
        winner = max(scores, key=scores.get)
        
        print(f"\n{date_str} → {winner}")
        print(f"  SRS={srs:+.3f}  SHS={shs:+.3f}  CLS={cls:.3f}  IPS={ips:+.3f}")
        for s, v in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            marker = ' ← GANADOR' if s == winner else ''
            print(f"    {s:<15} {v} puntos{marker}")
            
    except Exception as e:
        print(f"\n{date_str}: ERROR - {e}")

print("\n" + "=" * 70)
print("DIAGNÓSTICO COMPLETADO")
print("=" * 70)
