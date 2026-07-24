import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

df = pd.read_csv('outputs/backtest_v2_results.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()

import yfinance as yf
spy = yf.download('^GSPC', period='20y', auto_adjust=True)['Close']
spy_ret = spy.resample('W-FRI').last().pct_change().dropna()
spy_ret.name = 'spy_return'
common = df.index.intersection(spy_ret.index)
df = df.loc[common].copy()
df['spy_return'] = spy_ret.loc[common]

# Calcular métricas completas como referencia
stress_regimes = ['LIQUIDITY CRISIS','RECESSION','SLOWDOWN']
positive_regimes = ['EXPANSION','RECOVERY']
def calc_metrics(sub):
    acc = accuracy_score(sub['expected'], sub['obtained'])
    ret_stress = sub[sub['obtained'].isin(stress_regimes)]['spy_return']
    ret_pos = sub[sub['obtained'].isin(positive_regimes)]['spy_return']
    sharpe_stress = (ret_stress.mean()/(ret_stress.std()+1e-9))*np.sqrt(52) if len(ret_stress)>1 else np.nan
    sharpe_pos = (ret_pos.mean()/(ret_pos.std()+1e-9))*np.sqrt(52) if len(ret_pos)>1 else np.nan
    return acc, sharpe_stress, sharpe_pos

acc_all, ss_all, sp_all = calc_metrics(df)
print(f"Referencia completa: acc={acc_all:.2%}, sharpe_stress={ss_all:.2f}, sharpe_pos={sp_all:.2f}")
print("\n=== JACKKNIFE (eliminando cada año) ===")
for year in range(2007, 2026):
    mask = df.index.year == year
    if mask.sum() == 0:
        continue
    sub = df[~mask]
    acc, ss, sp = calc_metrics(sub)
    print(f"Sin {year}: acc={acc:.2%}, sharpe_stress={ss:.2f}, sharpe_pos={sp:.2f}")
