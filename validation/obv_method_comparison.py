# -*- coding: utf-8 -*-
# validation/obv_method_comparison.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
import numpy as np
from indicators.momentum import compute_obv
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
data = router.get_market_data(sectors, period='5y')

print('Comparación OBV.pct_change() vs OBV.diff()\n')
print('Sector | Correlación | Outliers pct_change | Outliers diff | % reducción outliers')
print('-------|-------------|---------------------|---------------|--------------------')

for sector in sectors:
    obv = compute_obv(data, sector)
    pct = obv.pct_change()
    dif = obv.diff()
    
    common = pct.index.intersection(dif.index)
    pct_c = pct.loc[common]
    dif_c = dif.loc[common]
    
    corr = pct_c.corr(dif_c)
    
    # Outliers: ±3 desviaciones estándar robustas (MAD)
    def count_outliers(series):
        median = series.median()
        mad = np.median(np.abs(series - median))
        threshold = 3 * 1.4826 * mad
        return (np.abs(series - median) > threshold).sum()
    
    outliers_pct = count_outliers(pct_c)
    outliers_dif = count_outliers(dif_c)
    reduccion = (1 - outliers_dif / outliers_pct) * 100 if outliers_pct > 0 else 0
    
    print(f'{sector:6s} | {corr:+.4f}     | {outliers_pct:5d}           | {outliers_dif:5d}        | {reduccion:+.1f}%')

print('\nConclusión: si la correlación es alta (>0.9) y diff reduce outliers, la alternativa es viable.')
