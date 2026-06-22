import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from regimes.financial_conditions import compute_liquidity_score
from regimes.volatility_regime import compute_volatility_regime
from regimes.macro_regime import compute_macro_regime
from regimes.sector_regime import compute_sector_scores
from src.utils import get_col
from src.macro_manual_loader import load_macro_manual
from config.tickers import MARKET_TICKERS

print("Cargando datos para correlación entre módulos...")
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
financial_list, volatility_list, macro_list, sector_list = [], [], [], []

for d in dates:
    dm = df[df.index <= d].copy()
    dmc = df_macro[df_macro['date'] <= d].copy() if df_macro is not None else None
    try:
        liq_s, _, _ = compute_liquidity_score(dm)
        financial_list.append(liq_s.iloc[-1])
        vix_r = get_col(dm, '^VIX', 'Close').pct_change(fill_method=None)
        vol_s, _, _ = compute_volatility_regime(vix_r)
        volatility_list.append(vol_s.iloc[-1])
        mac_s, _, _, _ = compute_macro_regime(dm, dmc, liq_s, vol_s)
        macro_list.append(mac_s.iloc[-1])
        sec = compute_sector_scores(dm)
        if sec and 'last_scores' in sec:
            sector_list.append(sec['last_scores'].mean())
        else:
            sector_list.append(np.nan)
    except:
        financial_list.append(np.nan)
        volatility_list.append(np.nan)
        macro_list.append(np.nan)
        sector_list.append(np.nan)

# Crear DataFrame y eliminar filas con NaN
df_mod = pd.DataFrame({
    'financial': financial_list,
    'volatility': volatility_list,
    'macro': macro_list,
    'sector': sector_list
}).dropna()

if len(df_mod) > 10:
    corr = df_mod.corr()
    print("\n=== CORRELACIÓN ENTRE MÓDULOS (mensual, 2015-2026) ===")
    print(corr.to_string(float_format=lambda x: f'{x:.2f}'))
    print("\nInterpretación: valores absolutos < 0.30 indican independencia entre capas.")
else:
    print("Datos insuficientes para calcular correlación entre módulos.")
