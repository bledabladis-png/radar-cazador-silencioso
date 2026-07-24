import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Cargar resultados del backtest
df = pd.read_csv('outputs/backtest_v2_results.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()

# Obtener retornos semanales del SPY
import yfinance as yf
spy = yf.download('^GSPC', period='20y', auto_adjust=True)['Close']
spy_ret = spy.resample('W-FRI').last().pct_change().dropna()
spy_ret.name = 'spy_return'

# Alinear fechas
common = df.index.intersection(spy_ret.index)
df = df.loc[common].copy()
df['spy_return'] = spy_ret.loc[common]

# Calcular retornos forward
for horizon in [1, 4, 12]:
    df[f'fwd_{horizon}w'] = df['spy_return'].rolling(horizon).apply(lambda x: np.prod(1+x)-1).shift(-horizon)

# Calcular IC para cada horizonte (solo fechas donde hay macro_score y fwd_return no nulo)
print("=== INFORMATION COEFFICIENT (Spearman) ===")
for horizon in [1, 4, 12]:
    mask = df['macro_score'].notna() & df[f'fwd_{horizon}w'].notna()
    if mask.sum() > 10:
        ic, pval = spearmanr(df.loc[mask, 'macro_score'], df.loc[mask, f'fwd_{horizon}w'])
        print(f"IC {horizon}sem: {ic:.4f} (p-value: {pval:.4f})")
    else:
        print(f"IC {horizon}sem: insuficientes datos ({mask.sum()} puntos)")
