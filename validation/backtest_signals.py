# -*- coding: utf-8 -*-
# validation/backtest_signals.py
# Fase 2: Backtest de senhales (IC con retornos futuros)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS
from indicators.momentum import compute_flow_proxy, compute_price_momentum
from indicators.wyckoff import wyckoff_score
from regimes.tactical_engine import compute_tactical_score
from regimes.structural_engine import compute_structural_score
from src.utils import get_col

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
benchmark = '^GSPC'
data = router.get_market_data(sectors + [benchmark], period='3y')

print('=== BACKTEST DE SEÑALES (IC SPEARMAN) ===')
print('Se calcula la correlación entre cada señal y el retorno futuro a 20d y 60d.\n')

results = []
for sector in sectors:
    close_full = get_col(data, sector, 'Close')
    # Retornos futuros (señal en fecha t, retorno de t a t+20/t+60)
    future20 = close_full.pct_change(20, fill_method=None).shift(-20)
    future60 = close_full.pct_change(60, fill_method=None).shift(-60)

    # Fechas de muestreo: cada 5 días, solo donde hay futuro válido y suficiente histórico
    valid_dates = future20.dropna().index.intersection(future60.dropna().index)
    sampled_dates = valid_dates[::5]
    if len(sampled_dates) < 30:
        print(f'{sector}: datos insuficientes')
        continue

    sig_rs, sig_mom, sig_flow, sig_wy, sig_tact, sig_struct = [], [], [], [], [], []
    fut20_list, fut60_list = [], []
    for fecha in sampled_dates:
        df_hasta = data.loc[:fecha]
        if len(df_hasta) < 200:
            continue
        try:
            close_s = get_col(df_hasta, sector, 'Close')
            close_b = get_col(df_hasta, benchmark, 'Close')
            # RS20
            rs = (close_s / close_b)
            rs20 = rs.pct_change(20, fill_method=None).iloc[-1]
            # Momentum20
            mom20 = close_s.pct_change(20, fill_method=None).iloc[-1]
            # Flow Proxy completo
            flow = compute_flow_proxy(df_hasta, sector).iloc[-1]
            # Wyckoff score
            wy = wyckoff_score(df_hasta, sector)[0].iloc[-1]
            # Tactical y Structural
            tac = compute_tactical_score(df_hasta, sector)
            struc = compute_structural_score(df_hasta, sector)

            sig_rs.append(rs20)
            sig_mom.append(mom20)
            sig_flow.append(flow)
            sig_wy.append(wy)
            sig_tact.append(tac)
            sig_struct.append(struc)
            fut20_list.append(future20.loc[fecha])
            fut60_list.append(future60.loc[fecha])
        except Exception:
            continue

    if len(sig_rs) < 30:
        print(f'{sector}: pocos datos válidos')
        continue

    df_sector = pd.DataFrame({
        'RS': sig_rs, 'Momentum': sig_mom, 'Flow': sig_flow,
        'Wyckoff': sig_wy, 'Tactical': sig_tact, 'Structural': sig_struct,
        'Fut20': fut20_list, 'Fut60': fut60_list
    }).dropna()

    ic20 = {c: spearmanr(df_sector[c], df_sector['Fut20']).correlation for c in ['RS','Momentum','Flow','Wyckoff','Tactical','Structural']}
    ic60 = {c: spearmanr(df_sector[c], df_sector['Fut60']).correlation for c in ['RS','Momentum','Flow','Wyckoff','Tactical','Structural']}

    results.append({'sector': sector, 'n': len(df_sector), **{f'{k}_IC20': v for k,v in ic20.items()}, **{f'{k}_IC60': v for k,v in ic60.items()}})
    print(f'{sector}: n={len(df_sector)}')
    print(f'  IC20: ' + ' | '.join([f'{k}:{v:.3f}' for k,v in ic20.items()]))
    print(f'  IC60: ' + ' | '.join([f'{k}:{v:.3f}' for k,v in ic60.items()]))

# Promedio
if results:
    df_res = pd.DataFrame(results)
    avg = {c: df_res[f'{c}_IC20'].mean() for c in ['RS','Momentum','Flow','Wyckoff','Tactical','Structural']}
    avg60 = {c: df_res[f'{c}_IC60'].mean() for c in ['RS','Momentum','Flow','Wyckoff','Tactical','Structural']}
    print('\n=== IC PROMEDIO ENTRE SECTORES ===')
    print('IC20:', avg)
    print('IC60:', avg60)
    df_res.to_csv('outputs/backtest_signals_results.csv', index=False)
    print('\nResultados guardados en outputs/backtest_signals_results.csv')
else:
    print('No se obtuvieron resultados.')
