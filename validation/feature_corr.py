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

print("Cargando datos para Feature Correlation...")
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

# Ultimo dia de cada mes para tener una serie manejable
dates = pd.date_range('2015-01-01', df.index[-1], freq='ME')
signals_list = []
for d in dates:
    dm = df[df.index <= d].copy()
    dmc = df_macro[df_macro['date'] <= d].copy() if df_macro is not None else None
    try:
        liq_s, _, _ = compute_liquidity_score(dm)
        vix_r = get_col(dm, '^VIX', 'Close').pct_change(fill_method=None)
        vol_s, _, _ = compute_volatility_regime(vix_r)
        _, _, _, sigs = compute_macro_regime(dm, dmc, liq_s, vol_s)
        if sigs is not None:
            signals_list.append(sigs.iloc[-1].to_dict())
    except:
        pass

if signals_list:
    df_signals = pd.DataFrame(signals_list)
    corr = df_signals.corr()
    print("\n=== MATRIZ DE CORRELACION ENTRE SEÑALES (mensual, 2015-2026) ===")
    print(corr.to_string(float_format=lambda x: f'{x:.2f}'))
    
    print("\n=== PARES ALTAMENTE CORRELACIONADOS (|r|>0.8) ===")
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i,j]) > 0.8:
                pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i,j]))
    if pairs:
        for p in pairs:
            print(f"{p[0]} <-> {p[1]}: {p[2]:.2f}")
    else:
        print("Ningun par supera 0.8. Señales suficientemente independientes.")
else:
    print("No se pudieron extraer señales.")
