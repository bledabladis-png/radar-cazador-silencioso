# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from indicators.wyckoff import trend_component, range_width, relative_volume, effort_vs_result
from src.utils import robust_zscore
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
data = router.get_market_data(sectors, period='5y')

# Cache de componentes normalizados
component_names = ['trend', 'range_width', 'rel_vol', 'effort']
default_weights = {'trend': 0.35, 'range_width': 0.25, 'rel_vol': 0.20, 'effort': 0.20}
cache = {s: {} for s in sectors}

for s in sectors:
    try:
        trend = trend_component(data, s).dropna()
        rw = range_width(data, s).dropna()
        rv = relative_volume(data, s).dropna()
        evr = effort_vs_result(data, s).dropna()
        common = trend.index.intersection(rw.index).intersection(rv.index).intersection(evr.index)
        if len(common) == 0:
            continue
        cache[s]['trend'] = np.tanh(robust_zscore(trend.loc[common]))
        cache[s]['range_width'] = -np.tanh(robust_zscore(rw.loc[common]))
        cache[s]['rel_vol'] = np.tanh(robust_zscore(rv.loc[common]))
        cache[s]['effort'] = np.tanh(robust_zscore(evr.loc[common]))
        # Guardar el último índice común real
        cache[s]['last_idx'] = common[-1]
    except Exception as e:
        print(f"Error cache {s}: {e}")

def compute_ranking(weights_dict):
    """Calcula el ranking de sectores con unos pesos dados (ya deben sumar 1)."""
    scores = {}
    for s, comps in cache.items():
        if not all(c in comps for c in weights_dict):
            continue
        last_idx = comps['last_idx']
        val = sum(weights_dict[c] * comps[c].loc[last_idx] for c in weights_dict)
        scores[s] = val
    ser = pd.Series(scores).dropna()
    return ser.rank(ascending=False) if len(ser) >= 5 else pd.Series()

# Ranking original con los 4 componentes
original_rank = compute_ranking(default_weights)
print(f"Modelo completo (4 componentes): sectores validos = {len(original_rank)}")

# ------------------------------------------------------------
# FASE 1: ABLACIÓN (sin variar pesos)
# ------------------------------------------------------------
print("\n=== ABLACION (pesos fijos renormalizados) ===")
ablation_results = {}
for removed in component_names:
    keep = [c for c in component_names if c != removed]
    # Pesos originales renormalizados
    w_renorm = {c: default_weights[c] for c in keep}
    total = sum(w_renorm.values())
    w_renorm = {c: v/total for c, v in w_renorm.items()}
    alt_rank = compute_ranking(w_renorm)
    if len(alt_rank) < 5:
        print(f"Sin {removed}: insuficientes sectores")
        continue
    common_idx = original_rank.index.intersection(alt_rank.index)
    tau, _ = kendalltau(original_rank.loc[common_idx], alt_rank.loc[common_idx])
    rank_change = (original_rank.loc[common_idx] - alt_rank.loc[common_idx]).abs().mean()
    ablation_results[removed] = {'tau': tau, 'rank_change': rank_change, 'weights': w_renorm}
    print(f"Sin {removed:12s}: Tau = {tau:.4f}, Cambio medio de ranking = {rank_change:.2f} posiciones")

# ------------------------------------------------------------
# FASE 2: MONTE CARLO SOBRE CADA SUBCONJUNTO
# ------------------------------------------------------------
print("\n=== MONTE CARLO POR SUBCONJUNTO (5000 simulaciones) ===")
rng = np.random.default_rng(42)
n_sims = 5000
all_simulations = []

for removed in component_names:
    keep = [c for c in component_names if c != removed]
    taus = []
    rank_changes = []
    for _ in range(n_sims):
        w = rng.dirichlet(np.ones(len(keep)))
        w_dict = dict(zip(keep, w))
        alt_rank = compute_ranking(w_dict)
        if len(alt_rank) < 5:
            continue
        common_idx = original_rank.index.intersection(alt_rank.index)
        if len(common_idx) < 5:
            continue
        tau, _ = kendalltau(original_rank.loc[common_idx], alt_rank.loc[common_idx])
        if np.isnan(tau):
            continue
        rank_change = (original_rank.loc[common_idx] - alt_rank.loc[common_idx]).abs().mean()
        taus.append(tau)
        rank_changes.append(rank_change)
        all_simulations.append({
            'removed': removed,
            'weights': w_dict,
            'tau': tau,
            'rank_change': rank_change
        })
    if taus:
        print(f"Sin {removed:12s}: Tau medio = {np.mean(taus):.4f}, std = {np.std(taus):.4f}, P5 = {np.percentile(taus, 5):.4f}, P95 = {np.percentile(taus, 95):.4f}, Cambio ranking medio = {np.mean(rank_changes):.2f}")
    else:
        print(f"Sin {removed:12s}: no se generaron simulaciones validas")

# Guardar resultados completos
if all_simulations:
    df = pd.DataFrame(all_simulations)
    df.to_csv('outputs/audit/wyckoff_ablation_results.csv', index=False)
    print("\nResultados guardados en outputs/audit/wyckoff_ablation_results.csv")
