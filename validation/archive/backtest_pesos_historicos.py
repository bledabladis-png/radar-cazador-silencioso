# -*- coding: utf-8 -*-
# backtest_pesos_historicos.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS
from indicators.wyckoff import trend_component, range_width, relative_volume_v41, effort_vs_result
from src.utils import robust_zscore

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
print("Descargando datos históricos (10 años)...")
data = router.get_market_data(sectors, period='10y')
print(f"Datos descargados: {len(data)} filas.")

# Pesos base
base_weights = np.array([0.35, 0.25, 0.20, 0.20])

# Variantes de pesos para probar
variants = {
    'original': base_weights,
    'iguales': np.array([0.25, 0.25, 0.25, 0.25]),
    'trend_dominante': np.array([0.55, 0.15, 0.15, 0.15]),
    'range_dominante': np.array([0.15, 0.55, 0.15, 0.15]),
    'sin_volume_effort': np.array([0.50, 0.50, 0.0, 0.0]),
    'solo_estructural': np.array([0.60, 0.40, 0.0, 0.0]),
}

# Calcular componentes normalizados para todo el histórico
print("Calculando componentes normalizados...")
components = {s: {} for s in sectors}
for s in sectors:
    trend = trend_component(data, s).dropna()
    rw = range_width(data, s).dropna()
    rv = relative_volume_v41(data, s).dropna()
    evr = effort_vs_result(data, s).dropna()
    common = trend.index.intersection(rw.index).intersection(rv.index).intersection(evr.index)
    components[s]['trend'] = np.tanh(robust_zscore(trend.loc[common]))
    components[s]['range'] = -np.tanh(robust_zscore(rw.loc[common]))
    components[s]['volume'] = np.tanh(robust_zscore(rv.loc[common]))
    components[s]['effort'] = np.tanh(robust_zscore(evr.loc[common]))

# Determinar fechas comunes a todos los sectores
fechas = None
for s in sectors:
    if fechas is None:
        fechas = components[s]['trend'].index
    else:
        fechas = fechas.intersection(components[s]['trend'].index)
fechas = fechas.sort_values()
print(f"Fechas comunes: {len(fechas)}")

# Evaluar estabilidad para cada variante
for nombre, pesos in variants.items():
    rankings_por_fecha = []
    for fecha in fechas:
        scores = {}
        for s in sectors:
            if fecha in components[s]['trend'].index:
                t = components[s]['trend'].loc[fecha]
                r = components[s]['range'].loc[fecha]
                v = components[s]['volume'].loc[fecha]
                e = components[s]['effort'].loc[fecha]
                scores[s] = pesos[0]*t + pesos[1]*r + pesos[2]*v + pesos[3]*e
        ser = pd.Series(scores).dropna()
        if len(ser) >= 5:
            rankings_por_fecha.append(ser.rank(ascending=False))

    # Kendall Tau entre rankings consecutivos
    taus_consecutivos = []
    for i in range(1, len(rankings_por_fecha)):
        r_prev = rankings_por_fecha[i-1]
        r_curr = rankings_por_fecha[i]
        common_idx = r_prev.index.intersection(r_curr.index)
        if len(common_idx) >= 5:
            tau, _ = kendalltau(r_prev.loc[common_idx], r_curr.loc[common_idx])
            if not np.isnan(tau):
                taus_consecutivos.append(tau)

    if taus_consecutivos:
        print(f"\nPesos: {nombre} {list(pesos)}")
        print(f"  Estabilidad temporal (Tau consecutivo medio): {np.mean(taus_consecutivos):.4f}")
        print(f"  Percentil 5: {np.percentile(taus_consecutivos, 5):.4f}")
    else:
        print(f"\nPesos: {nombre} - sin datos suficientes")
