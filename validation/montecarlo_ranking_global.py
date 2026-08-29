# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS
from regimes.tactical_engine import compute_tactical_score
from regimes.structural_engine import compute_structural_score

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
data = router.get_market_data(sectors, period='5y')

# Calcular scores reales para cada sector
tactical_scores = {}
structural_scores = {}
wyckoff_map = {'MARKUP': 0.75, 'ACCUMULATION': 0.5, 'RANGE': 0.0, 'DISTRIBUTION': -0.5}
for s in sectors:
    try:
        tactical_scores[s] = compute_tactical_score(data, s)
        structural_scores[s] = compute_structural_score(data, s)
    except Exception:
        tactical_scores[s] = 0.0
        structural_scores[s] = 0.0

# Ranking original con pesos por defecto (0.50 tactical + 0.50 structural, simplificado)
original = {}
for s in sectors:
    original[s] = 0.50 * tactical_scores.get(s, 0.0) + 0.50 * structural_scores.get(s, 0.0)
original_series = pd.Series(original).dropna()
original_rank = original_series.rank(ascending=False)
print(f"Sectores validos: {len(original_series)}")

# Monte Carlo variando el peso entre tactical y structural
rng = np.random.default_rng(42)
n_sims = 5000
taus = []
for _ in range(n_sims):
    w_tactical = rng.uniform(0, 1)
    w_structural = 1.0 - w_tactical
    alt = {}
    for s in sectors:
        alt[s] = w_tactical * tactical_scores.get(s, 0.0) + w_structural * structural_scores.get(s, 0.0)
    alt_series = pd.Series(alt).dropna()
    if len(alt_series) < 5:
        continue
    alt_rank = alt_series.rank(ascending=False)
    common_idx = original_rank.index.intersection(alt_rank.index)
    if len(common_idx) < 5:
        continue
    tau, _ = kendalltau(original_rank.loc[common_idx], alt_rank.loc[common_idx])
    if not np.isnan(tau):
        taus.append(tau)

if taus:
    print("Monte Carlo Tactical vs Structural (5000 simulaciones):")
    print(f"  Kendall Tau medio: {np.mean(taus):.4f}")
    print(f"  Kendall Tau minimo: {np.min(taus):.4f}")
    print(f"  Percentil 5: {np.percentile(taus, 5):.4f}")
    print(f"  Percentil 95: {np.percentile(taus, 95):.4f}")
else:
    print("No se generaron simulaciones validas.")
