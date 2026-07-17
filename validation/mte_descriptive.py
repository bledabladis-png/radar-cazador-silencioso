import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.utils import get_col
from indicators.mte import score_scenarios, sector_rotation_score, safe_haven_score, credit_stress_score, inflation_pressure_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VALIDACIÓN DESCRIPTIVA EX POST - MTE v1.0")
print("=" * 70)

# Cargar datos
df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
spy_close = get_col(df_market, 'SPY', 'Close')

# Fechas de evaluación (semanales, últimos 2 años)
end_date = df_market.index[-1]
start_date = end_date - pd.DateOffset(years=2)
eval_dates = pd.date_range(start_date, end_date, freq='W-FRI')
eval_dates = [d for d in eval_dates if d in df_market.index]

print(f"Evaluando {len(eval_dates)} semanas...")

rows = []
for i, date in enumerate(eval_dates):
    if i % 20 == 0:
        print(f"  Progreso: {i}/{len(eval_dates)}")
    idx = df_market.index.get_loc(date)
    df_slice = df_market.iloc[:idx+1]
    
    try:
        srs = sector_rotation_score(df_slice)
        shs = safe_haven_score(df_slice)
        # CLS aproximado
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
        
        # Retornos futuros del SPY
        spy_idx = spy_close.index.get_loc(date)
        ret_3m = spy_close.iloc[min(spy_idx+63, len(spy_close)-1)] / spy_close.iloc[spy_idx] - 1 if spy_idx+63 < len(spy_close) else np.nan
        ret_6m = spy_close.iloc[min(spy_idx+126, len(spy_close)-1)] / spy_close.iloc[spy_idx] - 1 if spy_idx+126 < len(spy_close) else np.nan
        
        rows.append({
            'date': date,
            'scenario': scenario,
            'ret_3m': ret_3m,
            'ret_6m': ret_6m
        })
    except:
        pass

df = pd.DataFrame(rows)
print(f"\nRegistros: {len(df)}")

# Tabla de retornos por escenario
print("\n" + "="*70)
print("RETORNOS DEL SPY POR ESCENARIO (validación descriptiva, no predictiva)")
print("="*70)

for scenario in ['EXPANSION', 'SOFT LANDING', 'RECESSION', 'STAGFLATION', 'CRISIS', 'MIXED']:
    subset = df[df['scenario'] == scenario]
    if len(subset) > 0:
        print(f"\n{scenario} (n={len(subset)}):")
        for col in ['ret_3m', 'ret_6m']:
            mean = subset[col].mean()
            median = subset[col].median()
            print(f"  {col}: media={mean*100:+.2f}%  mediana={median*100:+.2f}%")
    else:
        print(f"\n{scenario}: sin observaciones")

print("\n" + "="*70)
print("Nota: Esta validación es puramente descriptiva. No se utiliza para optimizar")
print("parámetros ni para generar señales de trading. Solo demuestra que los escenarios")
print("identificados por el MTE corresponden a comportamientos diferenciados del mercado.")
print("="*70)
