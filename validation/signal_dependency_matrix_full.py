# -*- coding: utf-8 -*-
# validation/signal_dependency_matrix_full.py (v2 - con Tactical/Structural muestreados)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
import numpy as np
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS
from indicators.momentum import compute_flow_proxy
from indicators.wyckoff import wyckoff_score
from regimes.tactical_engine import compute_tactical_score
from regimes.structural_engine import compute_structural_score

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
benchmark = '^GSPC'
data = router.get_market_data(sectors + [benchmark], period='2y')

# Fechas de muestreo: cada 5 días hábiles
fechas = data.index[::5]
print(f'Fechas de muestreo: {len(fechas)}')

# Para cada sector, construir diccionario de listas de senhales
signals = {s: {'RS': [], 'Momentum': [], 'Flow': [], 'Wyckoff': [], 'Tactical': [], 'Structural': []} for s in sectors}

print('Calculando senhales históricas (puede tardar un poco)...')
for fecha in fechas:
    df_hasta = data.loc[:fecha]
    if len(df_hasta) < 200:
        continue
    for sector in sectors:
        try:
            close = df_hasta.xs('Close', axis=1, level=0)[sector]
            close_spy = df_hasta.xs('Close', axis=1, level=0)[benchmark]
            rs = (close / close_spy).iloc[-1]
            rs_prev = (close.shift(20) / close_spy.shift(20)).iloc[-1]
            rs_mom = (rs - rs_prev) / rs_prev
            momentum = close.pct_change(20, fill_method=None).iloc[-1]
            flow = compute_flow_proxy(df_hasta, sector).iloc[-1]
            wy = wyckoff_score(df_hasta, sector)[0].iloc[-1]
            tactical = compute_tactical_score(df_hasta, sector)
            structural = compute_structural_score(df_hasta, sector)
            
            signals[sector]['RS'].append(rs_mom)
            signals[sector]['Momentum'].append(momentum)
            signals[sector]['Flow'].append(flow)
            signals[sector]['Wyckoff'].append(wy)
            signals[sector]['Tactical'].append(tactical)
            signals[sector]['Structural'].append(structural)
        except Exception:
            continue

# Promediar correlaciones entre sectores
pairs = [('RS','Momentum'), ('RS','Flow'), ('RS','Wyckoff'), ('RS','Tactical'), ('RS','Structural'),
         ('Momentum','Flow'), ('Momentum','Wyckoff'), ('Momentum','Tactical'), ('Momentum','Structural'),
         ('Flow','Wyckoff'), ('Flow','Tactical'), ('Flow','Structural'),
         ('Wyckoff','Tactical'), ('Wyckoff','Structural'), ('Tactical','Structural')]

all_corr = {pair: [] for pair in pairs}
for sector, sigdict in signals.items():
    df_sec = pd.DataFrame(sigdict).dropna()
    if len(df_sec) < 30:
        continue
    corr = df_sec.corr(method='spearman')
    for pair in pairs:
        if pair[0] in corr.columns and pair[1] in corr.columns:
            all_corr[pair].append(corr.loc[pair[0], pair[1]])

print('\nCorrelacion Spearman promedio entre senhales sectoriales (muestreo 5d, 2 anios):')
print('Par | Correlacion promedio | n_sectores')
print('----|-----------------------|-----------')
for pair, vals in sorted(all_corr.items()):
    if vals:
        print(f'{pair[0]:10s} vs {pair[1]:10s} | {np.mean(vals):+.4f} | {len(vals)}')
    else:
        print(f'{pair[0]:10s} vs {pair[1]:10s} | N/A | 0')
