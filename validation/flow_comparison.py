# -*- coding: utf-8 -*-
# validation/flow_comparison.py
# Fase C v4.3: Comparar flow simplificado (tactical_engine) vs flow proxy completo (momentum.py)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS
from indicators.momentum import compute_flow_proxy
from src.utils import get_col

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
benchmark = '^GSPC'
data = router.get_market_data(sectors + [benchmark], period='2y')

print('=== COMPARACION DE FLOW SIMPLIFICADO VS FLOW PROXY COMPLETO ===')
print('Muestreo cada 5 dias, 2 anios, 11 sectores\n')

results = []
for sector in sectors:
    close_full = get_col(data, sector, 'Close')
    future20 = close_full.pct_change(20, fill_method=None).shift(-20)
    future60 = close_full.pct_change(60, fill_method=None).shift(-60)
    valid_dates = future20.dropna().index.intersection(future60.dropna().index)
    sample_dates = valid_dates[::5]
    if len(sample_dates) < 30:
        continue

    simpl_vals, full_vals, fut20_list, fut60_list = [], [], [], []
    for fecha in sample_dates:
        df_hasta = data.loc[:fecha]
        if len(df_hasta) < 200:
            continue
        try:
            close_s = get_col(df_hasta, sector, 'Close')
            volume_s = get_col(df_hasta, sector, 'Volume')
            # Flow simplificado (replicando tactical_engine)
            if len(close_s) >= 6 and len(volume_s) >= 6:
                ret_5d = close_s.pct_change(5, fill_method=None).iloc[-1]
                vol_5d = volume_s.iloc[-5:].mean()
                vol_10d = volume_s.iloc[-10:].mean() if volume_s.iloc[-10:].mean() > 0 else 0.0
                flow_recent = ret_5d * vol_5d / vol_10d if vol_10d > 0 else 0.0
                flow_simpl_norm = np.tanh(flow_recent / 2) if pd.notna(flow_recent) else 0.0
            else:
                flow_simpl_norm = 0.0

            # Flow proxy completo
            flow_full_raw = compute_flow_proxy(df_hasta, sector).iloc[-1]
            flow_full_norm = np.tanh(flow_full_raw / 2) if pd.notna(flow_full_raw) else 0.0

            simpl_vals.append(flow_simpl_norm)
            full_vals.append(flow_full_norm)
            fut20_list.append(future20.loc[fecha])
            fut60_list.append(future60.loc[fecha])
        except Exception:
            continue

    if len(simpl_vals) < 30:
        continue

    df_sec = pd.DataFrame({
        'simplificado': simpl_vals,
        'completo': full_vals,
        'Fut20': fut20_list,
        'Fut60': fut60_list
    }).dropna()

    corr = spearmanr(df_sec['simplificado'], df_sec['completo']).correlation
    ic20_simpl = spearmanr(df_sec['simplificado'], df_sec['Fut20']).correlation
    ic60_simpl = spearmanr(df_sec['simplificado'], df_sec['Fut60']).correlation
    ic20_full = spearmanr(df_sec['completo'], df_sec['Fut20']).correlation
    ic60_full = spearmanr(df_sec['completo'], df_sec['Fut60']).correlation

    results.append({
        'sector': sector,
        'corr_simpl_full': corr,
        'ic20_simpl': ic20_simpl,
        'ic60_simpl': ic60_simpl,
        'ic20_full': ic20_full,
        'ic60_full': ic60_full,
        'mean_simpl': np.mean(df_sec['simplificado']),
        'std_simpl': np.std(df_sec['simplificado']),
        'mean_full': np.mean(df_sec['completo']),
        'std_full': np.std(df_sec['completo']),
        'n': len(df_sec)
    })
    print(f'{sector}: corr={corr:+.3f}, IC20_simpl={ic20_simpl:+.3f}, IC20_full={ic20_full:+.3f}, n={len(df_sec)}')

if results:
    df_res = pd.DataFrame(results)
    avg = df_res[['corr_simpl_full','ic20_simpl','ic60_simpl','ic20_full','ic60_full']].mean()
    print('\n=== PROMEDIOS ENTRE SECTORES ===')
    print(f'Correlacion simplificado vs completo: {avg["corr_simpl_full"]:+.3f}')
    print(f'IC20 simplificado: {avg["ic20_simpl"]:+.3f}, IC20 completo: {avg["ic20_full"]:+.3f}')
    print(f'IC60 simplificado: {avg["ic60_simpl"]:+.3f}, IC60 completo: {avg["ic60_full"]:+.3f}')

    df_res.to_csv('outputs/audit/flow_comparison_results.csv', index=False)
    with open('outputs/audit/auditoria_flow_comparison.md', 'w', encoding='utf-8') as f:
        f.write('# Fase C v4.3 - Comparacion de Flow simplificado vs Flow Proxy completo\n\n')
        f.write(f'**Fecha:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}\n\n')
        f.write('## Resultados por sector\n\n')
        f.write('| Sector | Corr simpl-full | IC20 simpl | IC20 full | IC60 simpl | IC60 full | n |\n')
        f.write('|--------|-----------------|------------|-----------|------------|-----------|---|\n')
        for _, row in df_res.iterrows():
            f.write(f"| {row['sector']} | {row['corr_simpl_full']:+.3f} | {row['ic20_simpl']:+.3f} | {row['ic20_full']:+.3f} | {row['ic60_simpl']:+.3f} | {row['ic60_full']:+.3f} | {row['n']} |\n")
        f.write(f"\n## Promedios\n\n")
        f.write(f"- Correlacion simplificado vs completo: **{avg['corr_simpl_full']:+.3f}**\n")
        f.write(f"- IC20 simplificado: {avg['ic20_simpl']:+.3f}, IC20 completo: {avg['ic20_full']:+.3f}\n")
        f.write(f"- IC60 simplificado: {avg['ic60_simpl']:+.3f}, IC60 completo: {avg['ic60_full']:+.3f}\n")
    print('\nResultados guardados en outputs/audit/flow_comparison_results.csv y outputs/audit/auditoria_flow_comparison.md')
else:
    print('No se obtuvieron resultados.')
