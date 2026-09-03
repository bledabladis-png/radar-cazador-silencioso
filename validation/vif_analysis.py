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
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("Cargando datos para VIF...")
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

# Obtener señales mensuales
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
    df_signals = pd.DataFrame(signals_list).dropna()
    print("\n=== VARIANCE INFLATION FACTOR (VIF) ===")
    print("VIF < 5: excelente | 5-10: revisar | >10: eliminar\n")
    vif_data = pd.DataFrame({
        'Variable': df_signals.columns,
        'VIF': [variance_inflation_factor(df_signals.values, i) for i in range(df_signals.shape[1])]
    }).sort_values('VIF', ascending=False)
    print(vif_data.to_string(index=False))
else:
    print("No se pudieron extraer señales.")
