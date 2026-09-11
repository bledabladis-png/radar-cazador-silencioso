import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from indicators.momentum import compute_price_momentum, compute_flow_proxy
from src.utils import get_col

# Descargar solo los 11 sectores (rápido)
sectors = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
data = yf.download(sectors, period='10y', auto_adjust=True)
if not isinstance(data.columns, pd.MultiIndex):
    data.columns = pd.MultiIndex.from_tuples(data.columns)

# Tomar último día de cada mes
dates = pd.date_range('2015-01-01', data.index[-1], freq='ME')
results = []
for d in dates:
    dm = data[data.index <= d].copy()
    if dm.empty:
        continue
    row = {'date': d}
    for sector in sectors:
        try:
            close = get_col(dm, sector, 'Close')
            mom = compute_price_momentum(dm, sector, 20).iloc[-1]
            flow = compute_flow_proxy(dm, sector).iloc[-1]
            row[f'{sector}_mom'] = mom
            row[f'{sector}_flow'] = flow
        except:
            row[f'{sector}_mom'] = np.nan
            row[f'{sector}_flow'] = np.nan
    results.append(row)

df = pd.DataFrame(results).set_index('date')
# Turnover: 1 - correlación de rankings entre meses consecutivos
mom_cols = [f'{s}_mom' for s in sectors]
flow_cols = [f'{s}_flow' for s in sectors]

def turnover_score(df, cols):
    ranks = df[cols].rank(axis=1, method='average')
    prev_ranks = ranks.shift(1)
    # Correlación de Spearman entre filas consecutivas
    taus = []
    for i in range(1, len(ranks)):
        r_now = ranks.iloc[i]
        r_prev = prev_ranks.iloc[i]
        mask = r_now.notna() & r_prev.notna()
        if mask.sum() > 2:
            tau = r_now[mask].corr(r_prev[mask], method='spearman')
            taus.append(1 - (tau + 1) / 2)  # transformar a [0,1], 0=igual, 1=cambio total
    return np.mean(taus) if taus else np.nan

print("=== TURNOVER DE RANKINGS SECTORIALES (mensual, 2015-2026) ===")
print(f"Momentum 20d turnover: {turnover_score(df, mom_cols):.2%}")
print(f"Flujo institucional turnover: {turnover_score(df, flow_cols):.2%}")
