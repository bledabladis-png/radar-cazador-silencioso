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
from config.weights import CRITICAL_WEIGHTS
from scipy.stats import spearmanr

print("Cargando datos para análisis de sensibilidad...")
tickers = []
for g in MARKET_TICKERS.values():
    if isinstance(g, dict): tickers.extend(g.values())
    elif isinstance(g, list): tickers.extend(g)
tickers = list(set(tickers))
df = yf.download(tickers, period='5y', auto_adjust=True)
if not isinstance(df.columns, pd.MultiIndex):
    df.columns = pd.MultiIndex.from_tuples(df.columns)

df_macro = load_macro_manual()
if df_macro is not None: df_macro['date'] = pd.to_datetime(df_macro['date'])

d = df.index[-1]
dm = df[df.index <= d].copy()
dmc = df_macro[df_macro['date'] <= d].copy() if df_macro is not None else None
try:
    liq_s, _, _ = compute_liquidity_score(dm)
    vix_r = get_col(dm, '^VIX', 'Close').pct_change(fill_method=None)
    vol_s, _, _ = compute_volatility_regime(vix_r)
    _, _, _, sigs = compute_macro_regime(dm, dmc, liq_s, vol_s)
except:
    print("Error al calcular señales.")
    exit()

if sigs is not None:
    # Usar las señales críticas disponibles (curve, credit, volatility, liquidity)
    available_keys = [k for k in ['curve', 'credit', 'volatility', 'liquidity'] if k in sigs.columns]
    if len(available_keys) < 2:
        print(f"Señales disponibles: {list(sigs.columns)}. Se necesitan al menos 2 señales críticas.")
        exit()
    base_weights = np.array([CRITICAL_WEIGHTS[k] for k in available_keys])
    base_weights = base_weights / base_weights.sum()
    signal_vals = np.array([sigs[k].iloc[-1] for k in available_keys])
    original_score = np.dot(base_weights, signal_vals)
    
    n_sim = 1000
    scores = []
    for _ in range(n_sim):
        noise = np.random.normal(0, 0.02, size=len(available_keys))
        weights = base_weights + noise
        weights = np.clip(weights, 0, None)
        weights = weights / weights.sum()
        scores.append(np.dot(weights, signal_vals))
    scores = np.array(scores)
    
    rho, _ = spearmanr(np.full(n_sim, original_score), scores)
    print(f"\n=== SENSIBILIDAD DE PESOS CRÍTICOS ===")
    print(f"Señales usadas: {available_keys}")
    print(f"Score original: {original_score:.4f}")
    print(f"Score medio tras ruido: {scores.mean():.4f}")
    print(f"Desviación estándar: {scores.std():.4f}")
    print(f"IC 95%: [{np.percentile(scores, 2.5):.4f}, {np.percentile(scores, 97.5):.4f}]")
    print(f"Correlación con original: {rho:.4f}")
    print(f"Estabilidad: {'ALTA' if rho > 0.95 else 'MEDIA' if rho > 0.90 else 'BAJA'}")
