import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from regimes.financial_conditions import compute_liquidity_score
from regimes.volatility_regime import compute_volatility_regime
from regimes.macro_regime import compute_macro_regime
from src.utils import get_col
from src.macro_manual_loader import load_macro_manual
from config.tickers import MARKET_TICKERS

print("Cargando datos para estabilidad temporal...")
tickers = []
for g in MARKET_TICKERS.values():
    if isinstance(g, dict): tickers.extend(g.values())
    elif isinstance(g, list): tickers.extend(g)
tickers = list(set(tickers))
df = yf.download(tickers, period='10y', auto_adjust=True)
if not isinstance(df.columns, pd.MultiIndex):
    df.columns = pd.MultiIndex.from_tuples(df.columns)

df_macro = load_macro_manual()
if df_macro is not None: df_macro['date'] = pd.to_datetime(df_macro['date'])

dates = pd.date_range('2015-01-01', df.index[-1], freq='ME')
signals_hist = []
for d in dates:
    dm = df[df.index <= d].copy()
    dmc = df_macro[df_macro['date'] <= d].copy() if df_macro is not None else None
    try:
        liq_s, _, _ = compute_liquidity_score(dm)
        vix_r = get_col(dm, '^VIX', 'Close').pct_change(fill_method=None)
        vol_s, _, _ = compute_volatility_regime(vix_r)
        _, _, _, sigs = compute_macro_regime(dm, dmc, liq_s, vol_s)
        if sigs is not None:
            row = sigs.iloc[-1].to_dict()
            row['date'] = d
            signals_hist.append(row)
    except:
        pass

if signals_hist:
    df_hist = pd.DataFrame(signals_hist).set_index('date')
    print("\n=== ESTABILIDAD TEMPORAL (Coeficiente de Variación Rolling 252d) ===")
    for col in df_hist.columns:
        rolling_mean = df_hist[col].rolling(252, min_periods=60).mean()
        rolling_std = df_hist[col].rolling(252, min_periods=60).std()
        cv = rolling_std / (rolling_mean.abs() + 1e-9)
        stability = 1 / (1 + cv.iloc[-1]) if not cv.empty and not np.isnan(cv.iloc[-1]) else np.nan
        print(f"  {col}: estabilidad = {stability:.3f}" if not np.isnan(stability) else f"  {col}: insuficientes datos")
else:
    print("No se pudieron extraer señales históricas.")
