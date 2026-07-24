import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

df = pd.read_csv('outputs/backtest_v2_results.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()

# Unir retornos del SPY
import yfinance as yf
spy = yf.download('^GSPC', period='20y', auto_adjust=True)['Close']
spy_ret = spy.resample('W-FRI').last().pct_change().dropna()
spy_ret.name = 'spy_return'
common = df.index.intersection(spy_ret.index)
df = df.loc[common].copy()
df['spy_return'] = spy_ret.loc[common]

# Ventanas de 3 años (156 semanas)
window = 156
start_years = range(2007, 2023)
results = []
for year in start_years:
    start = pd.Timestamp(f'{year}-01-01')
    end = start + pd.DateOffset(years=3)
    mask = (df.index >= start) & (df.index < end)
    sub = df.loc[mask]
    if len(sub) < 50:
        continue
    acc = accuracy_score(sub['expected'], sub['obtained'])
    # Sharpe bajo regímenes de estrés
    stress_regimes = ['LIQUIDITY CRISIS','RECESSION','SLOWDOWN']
    positive_regimes = ['EXPANSION','RECOVERY']
    ret_stress = sub[sub['obtained'].isin(stress_regimes)]['spy_return']
    ret_pos = sub[sub['obtained'].isin(positive_regimes)]['spy_return']
    sharpe_stress = (ret_stress.mean()/(ret_stress.std()+1e-9))*np.sqrt(52) if len(ret_stress)>1 else np.nan
    sharpe_pos = (ret_pos.mean()/(ret_pos.std()+1e-9))*np.sqrt(52) if len(ret_pos)>1 else np.nan
    results.append({'window': f'{year}-{year+2}', 'weeks': len(sub), 'accuracy': acc,
                    'sharpe_stress': sharpe_stress, 'sharpe_positive': sharpe_pos})
df_res = pd.DataFrame(results)
print("=== ROLLING WALK-FORWARD (ventanas de 3 años) ===")
print(df_res.to_string(float_format=lambda x: f'{x:.2f}'))
# Estabilidad: desviación estándar de las métricas
print(f"\nEstabilidad (std de accuracy): {df_res['accuracy'].std():.3f}")
print(f"Estabilidad (std sharpe_stress): {df_res['sharpe_stress'].std():.2f}")
print(f"Estabilidad (std sharpe_positive): {df_res['sharpe_positive'].std():.2f}")
