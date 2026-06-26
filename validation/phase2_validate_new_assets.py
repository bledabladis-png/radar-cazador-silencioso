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

# ============================================================
# CARGAR DATOS COMPLETOS (INCLUYENDO NUEVOS ACTIVOS)
# ============================================================
print("Descargando datos históricos...")
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

# ============================================================
# CALCULAR SEÑALES DEL SISTEMA Y RÉGIMEN MACRO (MENSUAL)
# ============================================================
print("Calculando regímenes macro...")
dates = pd.date_range('2015-01-01', df.index[-1], freq='ME')
results = []

for d in dates:
    dm = df[df.index <= d].copy()
    dmc = df_macro[df_macro['date'] <= d].copy() if df_macro is not None else None
    try:
        liq_s, _, _ = compute_liquidity_score(dm)
        vix_r = get_col(dm, '^VIX', 'Close').pct_change(fill_method=None)
        vol_s, _, _ = compute_volatility_regime(vix_r)
        _, regime, _, signals = compute_macro_regime(dm, dmc, liq_s, vol_s)
        row = {'date': d, 'regime': regime}
        # Añadir retornos de los nuevos activos en los últimos 20 días
        for t in ['VLUE','MTUM','QUAL','SCHC','EWX','EMB','ELD']:
            try:
                close = get_col(dm, t, 'Close')
                ret_1m = close.pct_change(20).iloc[-1]  # retorno mensual aprox
                row[f'{t}_ret'] = ret_1m
            except:
                row[f'{t}_ret'] = np.nan
        results.append(row)
    except:
        pass

df_res = pd.DataFrame(results).dropna()

# ============================================================
# 1. CORRELACIÓN CON SEÑALES EXISTENTES
# ============================================================
print("\n=== CORRELACIÓN DE NUEVOS ACTIVOS CON SEÑALES EXISTENTES ===")
# Calcular correlación de los retornos de los nuevos activos con las señales del sistema
# (market_strength, credit, etc.) en las mismas fechas
corr_data = pd.DataFrame()
for t in ['VLUE','MTUM','QUAL','SCHC','EWX','EMB','ELD']:
    corr_data[t] = df_res[f'{t}_ret']

# Añadir señales existentes (recalculamos para las mismas fechas)
signals_list = []
for d in df_res['date']:
    dm = df[df.index <= d].copy()
    dmc = df_macro[df_macro['date'] <= d].copy() if df_macro is not None else None
    try:
        liq_s, _, _ = compute_liquidity_score(dm)
        vix_r = get_col(dm, '^VIX', 'Close').pct_change(fill_method=None)
        vol_s, _, _ = compute_volatility_regime(vix_r)
        _, _, _, sigs = compute_macro_regime(dm, dmc, liq_s, vol_s)
        signals_list.append(sigs.iloc[-1].to_dict())
    except:
        signals_list.append({})

df_signals = pd.DataFrame(signals_list)
corr_data = pd.concat([corr_data, df_signals], axis=1).dropna()
corr_matrix = corr_data.corr()

# Mostrar correlación de los nuevos activos con las señales clave
for t in ['VLUE','MTUM','QUAL','SCHC','EWX','EMB','ELD']:
    print(f"\n{t}:")
    for signal in ['market_strength','credit','volatility','liquidity','curve','commodities','breadth']:
        if signal in corr_matrix.columns:
            r = corr_matrix.loc[t, signal]
            alert = " ⚠️ ALTA" if abs(r) > 0.7 else ""
            print(f"  vs {signal}: r={r:.3f}{alert}")

# ============================================================
# 2. CAPACIDAD DISCRIMINATORIA (SHARPE POR RÉGIMEN)
# ============================================================
print("\n=== SHARPE DE LOS NUEVOS ACTIVOS POR RÉGIMEN MACRO ===")
# Calcular retorno semanal de cada nuevo activo
weekly_prices = df.resample('W-FRI').last()
weekly_rets = weekly_prices.pct_change()

# Unir con regímenes (solo los viernes)
df_res['date'] = pd.to_datetime(df_res['date'])
weekly_regimes = df_res.set_index('date').resample('W-FRI').ffill()

for t in ['VLUE','MTUM','QUAL','SCHC','EWX','EMB','ELD']:
    print(f"\n{t}:")
    for regime in ['RECOVERY','EXPANSION','INFLATION SHOCK','LIQUIDITY CRISIS','RECESSION','SLOWDOWN']:
        mask = weekly_regimes['regime'] == regime
        ret = weekly_rets[('Close', t)].loc[mask].dropna()
        if len(ret) > 5:
            sharpe = ret.mean() / (ret.std() + 1e-9) * np.sqrt(52)
            print(f"  {regime}: semanas={len(ret)}, Sharpe={sharpe:.2f}")
