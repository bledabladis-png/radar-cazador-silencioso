import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
from regimes.financial_conditions import compute_liquidity_score
from regimes.volatility_regime import compute_volatility_regime
from scores.macro_scores import compute_macro_signals
from src.utils import get_col
from src.macro_manual_loader import load_macro_manual
from config.tickers import MARKET_TICKERS

print("Cargando datos para Lead‑Lag...")
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
signal_names = None
all_signals = []

for d in dates:
    dm = df[df.index <= d].copy()
    dmc = df_macro[df_macro['date'] <= d].copy() if df_macro is not None else None
    try:
        liq_s, _, _ = compute_liquidity_score(dm)
        vix_r = get_col(dm, '^VIX', 'Close').pct_change(fill_method=None)
        vol_s, _, _ = compute_volatility_regime(vix_r)
        signals = compute_macro_signals(dm, dmc, liq_s, vol_s)
        if signal_names is None:
            signal_names = list(signals.columns)
        all_signals.append(signals.iloc[-1].to_dict())
    except:
        pass

if not all_signals:
    print("No se pudieron extraer señales.")
    exit()

df_signals = pd.DataFrame(all_signals)

# Retornos del SPY alineados
spy_close = df[('Close', '^GSPC')].resample('ME').last().pct_change(fill_method=None).dropna()
df_signals['SPY_return'] = spy_close.reindex(df_signals.index)

print("\n=== LEAD‑LAG: CORRELACIÓN CRUZADA ===")
print("(Valor positivo = la señal antecede al SPY; negativo = el SPY antecede a la señal)\n")
for lag_weeks in [1, 2, 4, 8]:
    print(f"Desfase de {lag_weeks} semanas:")
    spy_shifted = df_signals['SPY_return'].shift(-lag_weeks)
    for col in signal_names:
        if col == 'real_liquidity':
            continue
        valid = df_signals[col].notna() & spy_shifted.notna()
        if valid.sum() > 10:
            corr, pval = pearsonr(df_signals.loc[valid, col], spy_shifted.loc[valid])
            direction = "→ SPY" if corr > 0 else "← SPY"
            if pval < 0.10:
                print(f"  {col}: corr={corr:.3f}, p={pval:.3f} {direction}")
    print()

print("\n=== CAUSALIDAD DE GRANGER (¿la señal causa al SPY?) ===")
for col in signal_names:
    if col == 'real_liquidity':
        continue
    data = df_signals[[col, 'SPY_return']].dropna()
    if len(data) > 30:
        try:
            result = grangercausalitytests(data, maxlag=4, verbose=False)
            p_values = [result[lag][0]['ssr_chi2test'][1] for lag in range(1, 5)]
            min_p = min(p_values)
            if min_p < 0.10:
                best_lag = p_values.index(min_p) + 1
                print(f"  {col} → SPY: p={min_p:.3f} (mejor lag={best_lag})")
        except:
            pass
