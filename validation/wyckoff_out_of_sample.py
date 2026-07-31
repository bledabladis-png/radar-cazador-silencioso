# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from indicators.wyckoff import (
    wyckoff_score, trend_component, range_width, 
    relative_volume_v41, effort_vs_result, robust_zscore
)
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
data = router.get_market_data(sectors, period='max')

periods = {
    '2016-2020': ('2016-01-01', '2020-01-01'),
    '2020-2023': ('2020-01-01', '2023-01-01'),
    '2023-2026': ('2023-01-01', '2026-08-01')
}

for period_name, (start, end) in periods.items():
    print(f"\n=== Periodo: {period_name} ===")
    mask = (data.index >= start) & (data.index <= end)
    period_data = data.loc[mask]
    if len(period_data) < 252:
        print("Datos insuficientes, omitiendo.")
        continue

    # Calcular scores originales con el ultimo dia del periodo
    scores = {}
    for s in sectors:
        try:
            result = wyckoff_score(period_data, s)
            score_series = result[0] if isinstance(result, tuple) else result
            scores[s] = score_series.dropna().iloc[-1] if not score_series.dropna().empty else np.nan
        except Exception as e:
            scores[s] = np.nan
    original = pd.Series(scores).dropna()
    if len(original) < 5:
        print("Menos de 5 sectores validos.")
        continue
    original_rank = original.rank(ascending=False)

    # Monte Carlo 1000 simulaciones
    rng = np.random.default_rng(42)
    taus = []
    for _ in range(1000):
        w = rng.dirichlet(np.ones(4))
        alt_scores = {}
        for s in sectors:
            try:
                trend = trend_component(period_data, s)
                rw = range_width(period_data, s)
                rv = relative_volume_v41(period_data, s)
                evr = effort_vs_result(period_data, s)
                common = trend.index.intersection(rw.index).intersection(rv.index).intersection(evr.index)
                if len(common) == 0:
                    continue
                t_n = np.tanh(robust_zscore(trend.loc[common])).iloc[-1]
                c_n = -np.tanh(robust_zscore(rw.loc[common])).iloc[-1]
                v_n = np.tanh(robust_zscore(rv.loc[common])).iloc[-1]
                e_n = np.tanh(robust_zscore(evr.loc[common])).iloc[-1]
                alt_scores[s] = w[0]*t_n + w[1]*c_n + w[2]*v_n + w[3]*e_n
            except:
                pass
        alt = pd.Series(alt_scores).dropna()
        if len(alt) < 5:
            continue
        alt_rank = alt.rank(ascending=False)
        common_idx = original_rank.index.intersection(alt_rank.index)
        if len(common_idx) < 5:
            continue
        tau, _ = kendalltau(original_rank.loc[common_idx], alt_rank.loc[common_idx])
        if not np.isnan(tau):
            taus.append(tau)
    if taus:
        print(f"Monte Carlo (simulaciones validas: {len(taus)})")
        print(f"  Tau medio: {np.mean(taus):.4f}, P5: {np.percentile(taus, 5):.4f}, P95: {np.percentile(taus, 95):.4f}")
    else:
        print("Monte Carlo sin resultados.")
