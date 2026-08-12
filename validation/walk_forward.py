# -*- coding: utf-8 -*-
# validation/walk_forward.py
# Fase A v4.3: Walk-forward out-of-sample (in-sample 2018-2024 vs OOS 2025-2026)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS
from indicators.momentum import compute_flow_proxy
from indicators.wyckoff import wyckoff_score
from regimes.tactical_engine import compute_tactical_score
from regimes.structural_engine import compute_structural_score
from src.utils import get_col

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
benchmark = '^GSPC'
period = '10y'
data = router.get_market_data(sectors + [benchmark], period=period)

# Periodos de evaluacion
IS_START, IS_END = '2018-01-01', '2024-12-31'
OOS_START, OOS_END = '2025-01-01', '2026-12-31'

def evaluate_period(data, start, end, label):
    print(f'Evaluando {label}...')
    mask = (data.index >= start) & (data.index <= end)
    period_data = data.loc[mask]
    close_full = {s: get_col(data, s, 'Close') for s in sectors}
    results = []
    for sector in sectors:
        close = close_full[sector]
        future20 = close.pct_change(20, fill_method=None).shift(-20)
        future60 = close.pct_change(60, fill_method=None).shift(-60)
        # fechas de muestreo dentro del periodo
        sample_dates = period_data.index[::5]
        sig_rs, sig_mom, sig_flow, sig_wy, sig_tact, sig_struct = [], [], [], [], [], []
        fut20_list, fut60_list = [], []
        for fecha in sample_dates:
            if fecha not in future20.index or fecha not in future60.index:
                continue
            df_hasta = data.loc[:fecha]
            if len(df_hasta) < 200:
                continue
            try:
                close_s = get_col(df_hasta, sector, 'Close')
                close_b = get_col(df_hasta, benchmark, 'Close')
                rs = (close_s / close_b)
                rs20 = rs.pct_change(20, fill_method=None).iloc[-1]
                mom20 = close_s.pct_change(20, fill_method=None).iloc[-1]
                flow = compute_flow_proxy(df_hasta, sector).iloc[-1]
                wy = wyckoff_score(df_hasta, sector)[0].iloc[-1]
                tact = compute_tactical_score(df_hasta, sector)
                struct = compute_structural_score(df_hasta, sector)
                sig_rs.append(rs20)
                sig_mom.append(mom20)
                sig_flow.append(flow)
                sig_wy.append(wy)
                sig_tact.append(tact)
                sig_struct.append(struct)
                fut20_list.append(future20.loc[fecha])
                fut60_list.append(future60.loc[fecha])
            except Exception:
                continue
        if len(sig_rs) < 30:
            continue
        df_sec = pd.DataFrame({
            'RS': sig_rs, 'Momentum': sig_mom, 'Flow': sig_flow,
            'Wyckoff': sig_wy, 'Tactical': sig_tact, 'Structural': sig_struct,
            'Fut20': fut20_list, 'Fut60': fut60_list
        }).dropna()
        if len(df_sec) < 20:
            continue
        ics20 = {c: spearmanr(df_sec[c], df_sec['Fut20']).correlation for c in ['RS','Momentum','Flow','Wyckoff','Tactical','Structural']}
        ics60 = {c: spearmanr(df_sec[c], df_sec['Fut60']).correlation for c in ['RS','Momentum','Flow','Wyckoff','Tactical','Structural']}
        results.append({'sector': sector, 'n': len(df_sec),
                        'IC20_RS': ics20['RS'], 'IC20_Momentum': ics20['Momentum'],
                        'IC20_Flow': ics20['Flow'], 'IC20_Wyckoff': ics20['Wyckoff'],
                        'IC20_Tactical': ics20['Tactical'], 'IC20_Structural': ics20['Structural'],
                        'IC60_RS': ics60['RS'], 'IC60_Momentum': ics60['Momentum'],
                        'IC60_Flow': ics60['Flow'], 'IC60_Wyckoff': ics60['Wyckoff'],
                        'IC60_Tactical': ics60['Tactical'], 'IC60_Structural': ics60['Structural']})
    if not results:
        return None
    df_res = pd.DataFrame(results)
    avg = {}
    for c in ['RS','Momentum','Flow','Wyckoff','Tactical','Structural']:
        avg[f'IC20_{c}'] = df_res[f'IC20_{c}'].mean()
        avg[f'IC60_{c}'] = df_res[f'IC60_{c}'].mean()
    return {'label': label, 'n_sectors': len(df_res), 'avg': avg, 'detail': df_res}

print('Iniciando walk-forward...')
is_results = evaluate_period(data, IS_START, IS_END, 'In-sample 2018-2024')
oos_results = evaluate_period(data, OOS_START, OOS_END, 'Out-of-sample 2025-2026')

if is_results and oos_results:
    print('\n=== RESUMEN WALK-FORWARD ===')
    signals = ['RS','Momentum','Flow','Wyckoff','Tactical','Structural']
    print('Senhal | IC20 IS | IC20 OOS | IC60 IS | IC60 OOS')
    for s in signals:
        is20 = is_results['avg'][f'IC20_{s}']
        oos20 = oos_results['avg'][f'IC20_{s}']
        is60 = is_results['avg'][f'IC60_{s}']
        oos60 = oos_results['avg'][f'IC60_{s}']
        print(f'{s:12s} | {is20:+.3f} | {oos20:+.3f} | {is60:+.3f} | {oos60:+.3f}')

    # Guardar CSV
    pd.concat([is_results['detail'], oos_results['detail']], keys=['IS','OOS']).to_csv('outputs/audit/walk_forward_ics.csv')
    # Informe markdown
    with open('outputs/audit/walk_forward_results.md', 'w', encoding='utf-8') as f:
        f.write('# Walk-Forward Out-of-Sample\n\n')
        f.write(f'In-sample: {IS_START} a {IS_END}\n')
        f.write(f'Out-of-sample: {OOS_START} a {OOS_END}\n\n')
        f.write('| Senhal | IC20 IS | IC20 OOS | IC60 IS | IC60 OOS |\n')
        f.write('|--------|---------|----------|---------|----------|\n')
        for s in signals:
            is20 = is_results['avg'][f'IC20_{s}']
            oos20 = oos_results['avg'][f'IC20_{s}']
            is60 = is_results['avg'][f'IC60_{s}']
            oos60 = oos_results['avg'][f'IC60_{s}']
            f.write(f'| {s} | {is20:+.3f} | {oos20:+.3f} | {is60:+.3f} | {oos60:+.3f} |\n')
    print('\nResultados guardados en outputs/audit/walk_forward_results.md y outputs/audit/walk_forward_ics.csv')
else:
    print('No se pudieron obtener resultados para ambos periodos.')
