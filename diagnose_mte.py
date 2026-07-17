import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.utils import get_col
from indicators.mte import score_scenarios, sector_rotation_score, safe_haven_score, credit_stress_score, inflation_pressure_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DIAGNÓSTICO DE CLASIFICACIÓN - MTE v1.0")
print("=" * 70)

df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)

# Fechas clave para diagnosticar
test_dates = [
    '2020-03-23',  # COVID low
    '2020-03-16',  # COVID crisis week
    '2022-06-15',  # Inflation peak
    '2023-07-31',  # Recovery 2023
    '2024-01-02',  # Early 2024
]

for date_str in test_dates:
    if date_str not in df_market.index:
        print(f"\n{date_str}: fecha no disponible en market_data.csv")
        continue
    
    idx = df_market.index.get_loc(date_str)
    df_slice = df_market.iloc[:idx+1]
    
    try:
        srs = sector_rotation_score(df_slice)
        shs = safe_haven_score(df_slice)
        # CLS aproximado
        vix_close = get_col(df_slice, '^VIX', 'Close')
        vix_ret = vix_close.pct_change().rolling(20).std().iloc[-1]
        vix_ma = vix_close.pct_change().rolling(60).std().mean()
        fc_approx = float(np.clip(np.tanh((vix_ret / (vix_ma + 1e-9) - 1) / 2), 0, 1))
        hyg = get_col(df_slice, 'HYG', 'Close')
        lqd = get_col(df_slice, 'LQD', 'Close')
        spread = hyg / lqd
        cred_approx = float(np.clip(np.tanh((1/spread.iloc[-1] - 1) / 2), 0, 1))  # nivel del spread
        cls = float(np.mean([fc_approx, cred_approx, 0.3, 0.3]))
        ips = inflation_pressure_score(df_slice)
        
        scores = score_scenarios(srs, shs, cls, ips)
        winner = max(scores, key=scores.get)
        
        print(f"\n{date_str} → {winner}")
        print(f"  SRS={srs:+.3f}  SHS={shs:+.3f}  CLS={cls:.3f}  IPS={ips:+.3f}")
        for s, v in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            marker = ' ← GANADOR' if s == winner else ''
            print(f"    {s:<15} {v} puntos{marker}")
            
    except Exception as e:
        print(f"\n{date_str}: ERROR - {e}")

print("\n" + "="*70)
print("DIAGNÓSTICO COMPLETADO")
print("="*70)
