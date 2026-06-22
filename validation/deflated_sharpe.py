import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm

# Cargar resultados del backtest
df = pd.read_csv('outputs/backtest_v3_results.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()

# Retornos del SPY
spy = yf.download('^GSPC', period='20y', auto_adjust=True)['Close']
spy_ret = spy.resample('W-FRI').last().pct_change().dropna()
spy_ret.name = 'spy_return'

# Unir
common = df.index.intersection(spy_ret.index)
df = df.loc[common].copy()
df['spy_return'] = spy_ret.loc[common]

print("=== DEFLATED SHARPE RATIO POR RÉGIMEN ===")
print("(DSR > 0.95 se considera significativo)\n")

regimes = df['obtained'].unique()
sharpe_list = []

for regime in sorted(regimes):
    mask = df['obtained'] == regime
    rets = df.loc[mask, 'spy_return'].dropna()
    n = len(rets)
    if n < 10:
        continue
    sr = rets.mean() / (rets.std() + 1e-9) * np.sqrt(52)
    sharpe_list.append((regime, sr, n))

# Calcular E[Sharpe] bajo la hipótesis nula (ruido)
all_rets = df['spy_return'].dropna().values
n_sim = 1000
null_sharpes = []
for _ in range(n_sim):
    sample = np.random.choice(all_rets, size=120, replace=True)
    sr = sample.mean() / (sample.std() + 1e-9) * np.sqrt(52)
    null_sharpes.append(sr)

null_mean = np.mean(null_sharpes)
null_std = np.std(null_sharpes)

for regime, sr, n in sharpe_list:
    z = (sr - null_mean) / (null_std + 1e-9)
    dsr = norm.cdf(z)
    sig = "SIGNIFICATIVO" if dsr > 0.95 else "no significativo"
    print(f"  {regime}: Sharpe={sr:.2f}, n={n}, DSR={dsr:.3f} ({sig})")
