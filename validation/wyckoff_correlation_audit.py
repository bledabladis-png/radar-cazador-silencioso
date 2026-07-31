# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from indicators.wyckoff import trend_component, range_width, relative_volume, effort_vs_result
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS

router = DataRouter()
sectors = MARKET_TICKERS['sectors']   # lista de tickers, ej: ['XLK','XLF',...]
data = router.get_market_data(sectors, period='5y')

components = ['trend', 'range_width', 'rel_vol', 'effort']
series = {c: [] for c in components}

for ticker in sectors:
    try:
        trend = trend_component(data, ticker).dropna()
        rw = range_width(data, ticker).dropna()
        rv = relative_volume(data, ticker).dropna()
        evr = effort_vs_result(data, ticker).dropna()
        common = trend.index.intersection(rw.index).intersection(rv.index).intersection(evr.index)
        series['trend'].extend(trend.loc[common].values)
        series['range_width'].extend(rw.loc[common].values)
        series['rel_vol'].extend(rv.loc[common].values)
        series['effort'].extend(evr.loc[common].values)
    except Exception as e:
        print(f"Error en {ticker}: {e}")

df = pd.DataFrame(series)
print("Pearson:")
print(df.corr(method='pearson').to_string())
print("\nSpearman:")
print(df.corr(method='spearman').to_string())

# Bootstrap
def bootstrap_spearman(x, y, n=1000):
    rng = np.random.default_rng(42)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        rho, _ = spearmanr(x[idx], y[idx])
        values.append(rho)
    return np.mean(values), np.percentile(values, 2.5), np.percentile(values, 97.5)

pairs = [('trend', 'effort'), ('trend', 'range_width'), ('trend', 'rel_vol'),
         ('range_width', 'rel_vol'), ('range_width', 'effort'), ('rel_vol', 'effort')]
print("\nBootstrap Spearman (95% CI):")
for a, b in pairs:
    mean, low, high = bootstrap_spearman(df[a].dropna(), df[b].dropna())
    print(f"  {a} vs {b}: {mean:.3f} [{low:.3f}, {high:.3f}]")
