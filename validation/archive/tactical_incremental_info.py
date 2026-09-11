# -*- coding: utf-8 -*-
# validation/tactical_incremental_info.py
# Fase 1: Analisis de informacion incremental del Tactical Score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
import numpy as np
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS
from src.utils import get_col

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
benchmark = '^GSPC'
data = router.get_market_data(sectors + [benchmark], period='2y')

# Orden de inclusión según pesos del Tactical Score
ORDER = ['rs20_norm', 'mom20_norm', 'flow_norm', 'breadth_norm', 'accel_norm']
COMPONENT_NAMES = {
    'rs20_norm': 'RS20',
    'mom20_norm': 'Momentum20',
    'flow_norm': 'Flow',
    'breadth_norm': 'Breadth20',
    'accel_norm': 'Acceleration'
}
WEIGHTS = {
    'rs20': 0.30,
    'momentum20': 0.25,
    'flow_recent': 0.20,
    'breadth20': 0.15,
    'acceleration': 0.10
}

def r2_score(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0

# Para cada sector, construir series de componentes normalizados y score
sector_results = []

for sector in sectors:
    comps = {c: [] for c in ORDER}
    scores = []
    fechas_muestreo = data.index[::5]
    for fecha in fechas_muestreo:
        df_hasta = data.loc[:fecha]
        if len(df_hasta) < 200:
            continue
        try:
            close_sector = get_col(df_hasta, sector, 'Close')
            close_bench = get_col(df_hasta, benchmark, 'Close')
            volume_sector = get_col(df_hasta, sector, 'Volume')

            # RS20
            rs = close_sector / close_bench
            rs20 = rs.pct_change(20, fill_method=None).iloc[-1] if len(rs) >= 21 else np.nan
            rs20_norm = np.tanh(rs20 * 10) if pd.notna(rs20) else np.nan

            # Momentum20
            mom20 = close_sector.pct_change(20, fill_method=None).iloc[-1] if len(close_sector) >= 21 else np.nan
            mom20_norm = np.tanh(mom20 * 5) if pd.notna(mom20) else np.nan

            # Flow reciente
            if len(close_sector) >= 6 and len(volume_sector) >= 6:
                ret_5d = close_sector.pct_change(5, fill_method=None).iloc[-1]
                vol_5d = volume_sector.iloc[-5:].mean()
                flow_recent = ret_5d * vol_5d / volume_sector.iloc[-10:].mean() if volume_sector.iloc[-10:].mean() > 0 else np.nan
                flow_norm = np.tanh(flow_recent / 2) if pd.notna(flow_recent) else np.nan
            else:
                flow_norm = np.nan

            # Breadth20
            if len(close_sector) >= 20:
                ema20 = close_sector.ewm(span=20, min_periods=20).mean()
                breadth20 = (close_sector.iloc[-20:] > ema20.iloc[-20:]).sum() / 20
                breadth_norm = (breadth20 - 0.5) * 2
            else:
                breadth_norm = np.nan

            # Acceleration
            if len(close_sector) >= 26:
                mom20_prev = close_sector.pct_change(20, fill_method=None).iloc[-6]
                accel = (mom20 - mom20_prev) * 5
                accel_norm = np.tanh(accel * 3) if pd.notna(accel) else np.nan
            else:
                accel_norm = np.nan

            comps['rs20_norm'].append(rs20_norm)
            comps['mom20_norm'].append(mom20_norm)
            comps['flow_norm'].append(flow_norm)
            comps['breadth_norm'].append(breadth_norm)
            comps['accel_norm'].append(accel_norm)

            # Score lineal (sin clipping para preservar la relacion lineal)
            score = (WEIGHTS['rs20']*rs20_norm + WEIGHTS['momentum20']*mom20_norm +
                     WEIGHTS['flow_recent']*flow_norm + WEIGHTS['breadth20']*breadth_norm +
                     WEIGHTS['acceleration']*accel_norm)
            scores.append(score)
        except Exception:
            continue

    df_sec = pd.DataFrame(comps)
    df_sec['score'] = scores
    df_sec = df_sec.dropna()
    if len(df_sec) < 30:
        continue

    # Analisis incremental
    r2_acc = []
    X_selected = None
    y = df_sec['score'].values
    for c in ORDER:
        if X_selected is None:
            X_selected = df_sec[[c]].values
        else:
            X_selected = np.column_stack([X_selected, df_sec[c].values])
        r2 = r2_score(y, X_selected)
        r2_acc.append(r2)
    sector_results.append({'sector': sector, 'r2_acc': r2_acc, 'n': len(df_sec)})

# Mostrar resultados
print('Incremento de R2 por componente (orden: RS20 -> Momentum20 -> Flow -> Breadth20 -> Acceleration):')
print('Sector | R2_RS20 | +Mom20 | +Flow | +Breadth | +Accel | n')
print('-------|---------|--------|-------|----------|--------|----')
for res in sector_results:
    r = res['r2_acc']
    print(f"{res['sector']:6s} | {r[0]:.4f} | {r[1]-r[0]:.4f} | {r[2]-r[1]:.4f} | {r[3]-r[2]:.4f} | {r[4]-r[3]:.4f} | {res['n']}")

# Promedio
avg_r2 = np.mean([res['r2_acc'] for res in sector_results], axis=0)
print('\nPromedio:')
print(f"RS20: {avg_r2[0]:.4f}, +Momentum20: {avg_r2[1]-avg_r2[0]:.4f}, +Flow: {avg_r2[2]-avg_r2[1]:.4f}, +Breadth20: {avg_r2[3]-avg_r2[2]:.4f}, +Acceleration: {avg_r2[4]-avg_r2[3]:.4f}")
