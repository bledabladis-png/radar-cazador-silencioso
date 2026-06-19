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

print("Cargando datos para Feature Importance...")
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

# Fechas de evaluación (último día de cada mes desde 2015)
dates = pd.date_range('2015-01-01', df.index[-1], freq='ME')
results = []
for d in dates:
    dm = df[df.index <= d].copy()
    dmc = df_macro[df_macro['date'] <= d].copy() if df_macro is not None else None
    try:
        liq_s, _, _ = compute_liquidity_score(dm)
        vix_r = get_col(dm, '^VIX', 'Close').pct_change(fill_method=None)
        vol_s, _, _ = compute_volatility_regime(vix_r)
        _, _, _, sigs = compute_macro_regime(dm, dmc, liq_s, vol_s)
        if sigs is not None:
            results.append(sigs.iloc[-1].to_dict())
    except:
        pass

df_signals = pd.DataFrame(results)
# Calcular macro_score base (mismo cálculo que en el modelo, simplificado)
# Usaremos la media de todas las señales como score simplificado para la importancia
base_score = df_signals.mean(axis=1)

print("\n=== FEATURE IMPORTANCE (Permutation) ===")
print("Mide cuánto empeora la correlación consigo mismo al permutar cada señal.")
importances = {}
for col in df_signals.columns:
    # Permutar la columna y recalcular score
    df_perm = df_signals.copy()
    df_perm[col] = np.random.permutation(df_perm[col].values)
    perm_score = df_perm.mean(axis=1)
    # Importancia = 1 - correlación entre score original y permutado (mayor => más importante)
    ic = base_score.corr(perm_score)
    importance = 1 - abs(ic)
    importances[col] = importance
    print(f"{col}: importancia = {importance:.4f}")

# Ordenar por importancia
sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
print("\nOrdenadas:")
for col, imp in sorted_imp:
    print(f"{col}: {imp:.4f}")
