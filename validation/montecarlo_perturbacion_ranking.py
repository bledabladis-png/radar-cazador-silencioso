# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import kendalltau

df = pd.read_csv('outputs/report/sector_rankings.csv')
original_scores = df.set_index('ticker')['score']
original_rank = original_scores.rank(ascending=False)

rng = np.random.default_rng(42)
n_sims = 5000
taus = []
for _ in range(n_sims):
    noise = rng.normal(0, 0.02, len(original_scores))  # ruido gaussiano con std=0.02
    perturbed = original_scores + noise
    perturbed_rank = perturbed.rank(ascending=False)
    tau, _ = kendalltau(original_rank, perturbed_rank)
    if not np.isnan(tau):
        taus.append(tau)

print("Monte Carlo de perturbacion del ranking global (5000 simulaciones):")
print("  Ruido: N(0, 0.02)")
print(f"  Kendall Tau medio: {np.mean(taus):.4f}")
print(f"  Kendall Tau minimo: {np.min(taus):.4f}")
print(f"  Percentil 5: {np.percentile(taus, 5):.4f}")
print(f"  Percentil 95: {np.percentile(taus, 95):.4f}")
