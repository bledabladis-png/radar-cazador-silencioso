import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from regimes.financial_conditions import compute_liquidity_score
from regimes.volatility_regime import compute_volatility_regime
from regimes.macro_regime import compute_macro_regime
from src.utils import get_col
from src.macro_manual_loader import load_macro_manual
from config.tickers import MARKET_TICKERS

EVENTS = {
    '2008-09-15': 'Lehman Brothers',
    '2020-03-23': 'COVID low',
    '2022-06-15': 'Inflation peak',
    '2023-07-31': 'Recovery 2023',
    '2025-01-02': 'Early 2025'
}

print("Cargando datos...")
tickers = []
for g in MARKET_TICKERS.values():
    if isinstance(g, dict): tickers.extend(g.values())
    elif isinstance(g, list): tickers.extend(g)
tickers = list(set(tickers))
df = yf.download(tickers, period='20y', auto_adjust=True)
if not isinstance(df.columns, pd.MultiIndex):
    df.columns = pd.MultiIndex.from_tuples(df.columns)

df_macro = load_macro_manual()
if df_macro is not None: df_macro['date'] = pd.to_datetime(df_macro['date'])

print("\n=== ANÁLISIS POR EVENTOS ===")
for date_str, label in EVENTS.items():
    d = pd.Timestamp(date_str)
    dm = df[df.index <= d].copy()
    dmc = df_macro[df_macro['date'] <= d].copy() if df_macro is not None else None
    try:
        liq_s, liq_r, _ = compute_liquidity_score(dm)
        vix_r = get_col(dm, '^VIX', 'Close').pct_change(fill_method=None)
        vol_s, vol_r, _ = compute_volatility_regime(vix_r)
        _, macro_r, conf, sigs = compute_macro_regime(dm, dmc, liq_s, vol_s)
        print(f"{date_str} ({label}): régimen={macro_r}, confianza={conf:.0%}")
    except Exception as e:
        print(f"{date_str} ({label}): ERROR {e}")
