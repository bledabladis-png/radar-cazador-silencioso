# -*- coding: utf-8 -*-
# validation/signal_dependency_matrix.py (v3)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
import numpy as np
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS
from indicators.wyckoff import wyckoff_score
from indicators.momentum import compute_price_momentum, compute_flow_proxy
from indicators.breadth import compute_breadth
from indicators.darkpool import compute_darkpool_signals

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
data = router.get_market_data(sectors, period='2y')

# Recolectar señales en listas
rs_vals, flow_vals, wyckoff_vals, ats_vals = [], [], [], []

# Breadth (global, sin argumento sectors)
b20, b50, b200, nh, nl = compute_breadth(data)

# ATS (global)
dp = compute_darkpool_signals()
ats_global = dp['media_dark_pool'] if dp else 0

for s in sectors:
    try:
        # RS = retorno 20d
        rs = compute_price_momentum(data, s).iloc[-1]
        # Flow
        flow = compute_flow_proxy(data, s).iloc[-1]
        # Wyckoff
        wy = wyckoff_score(data, s)[0].iloc[-1]
        
        rs_vals.append(rs)
        flow_vals.append(flow)
        wyckoff_vals.append(wy)
        ats_vals.append(ats_global)
    except:
        pass

df = pd.DataFrame({
    'RS': rs_vals,
    'Flow': flow_vals,
    'Wyckoff': wyckoff_vals,
    'ATS': ats_vals
}).dropna()

# Añadir breadth (mismo valor para todos los sectores)
if len(df) > 0:
    df['Breadth'] = b20.iloc[-1]

print('Pearson:')
print(df.corr(method='pearson').round(4).to_string())
print('\nSpearman:')
print(df.corr(method='spearman').round(4).to_string())
