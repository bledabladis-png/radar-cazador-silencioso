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

# Calcular componentes normalizados en la última fecha común para cada sector
scores_original = {}
component_cache = {}  # guardamos los arrays normalizados para no recalcular 5000 veces
for s in sectors:
    try:
        trend = trend_component(data, s).dropna()
        rw = range_width(data, s).dropna()
        rv = relative_volume(data, s).dropna()
        evr = effort_vs_result(data, s).dropna()
        common = trend.index.intersection(rw.index).intersection(rv.index).intersection(evr.index)
        if len(common) == 0:
            continue
        t_norm = np.tanh(robust_zscore(trend.loc[common]))
        c_norm = -np.tanh(robust_zscore(rw.loc[common]))
        v_norm = np.tanh(robust_zscore(rv.loc[common]))
        e_norm = np.tanh(robust_zscore(evr.loc[common]))
        component_cache[s] = (t_norm, c_norm, v_norm, e_norm)
        last_idx = common[-1]
        scores_original[s] = 0.35*t_norm.loc[last_idx] + 0.25*c_norm.loc[last_idx] + 0.20*v_norm.loc[last_idx] + 0.20*e_norm.loc[last_idx]
    except Exception as e:
        print(f"Error original {s}: {e}")

original = pd.Series(scores_original).dropna()
original_rank = original.rank(ascending=False)
print(f"Sectores con score original: {len(original)}")

rng = np.random.default_rng(42)
results = []
taus = []
for i in range(5000):
    w = rng.dirichlet(np.ones(4))
    scores_alt = {}
    for s, (t_norm, c_norm, v_norm, e_norm) in component_cache.items():
        try:
            last_idx = t_norm.index[-1]
            scores_alt[s] = w[0]*t_norm.loc[last_idx] + w[1]*c_norm.loc[last_idx] + w[2]*v_norm.loc[last_idx] + w[3]*e_norm.loc[last_idx]
        except Exception as e:
            print(f"Error alt {s}: {e}")
            scores_alt[s] = np.nan

    alt = pd.Series(scores_alt).dropna()
    if len(alt) < 5:
        continue

    # Alinear índices con concat + dropna
    common = pd.concat([original_rank, alt.rank(ascending=False)], axis=1, keys=['orig', 'alt']).dropna()
    if len(common) < 5:
        continue

    tau, pval = kendalltau(common['orig'], common['alt'])
    if np.isnan(tau):
        continue

    taus.append(tau)
    results.append({'weights': w.tolist(), 'tau': tau, 'scores': scores_alt.copy()})

if taus:
    print(f"Monte Carlo (simulaciones validas: {len(taus)})")
    print(f"  Kendall Tau medio: {np.mean(taus):.4f}")
    print(f"  Kendall Tau minimo: {np.min(taus):.4f}")
    print(f"  Percentil 5: {np.percentile(taus, 5):.4f}")
    print(f"  Percentil 95: {np.percentile(taus, 95):.4f}")
    # Guardar resultados para análisis futuro
    pd.DataFrame(results).to_csv('outputs/wyckoff_montecarlo_results.csv', index=False)
    print("Resultados guardados en outputs/wyckoff_montecarlo_results.csv")
else:
    print("No se generaron simulaciones validas.")
