# global_cross_region_v4.py – Cross-Region Module (alineación global)
# Mide correlación y coherencia entre regiones usando retornos,
# no PWM. Independiente de Flow y Risk.

import pandas as pd
import numpy as np

def compute_region_alignment(weekly_returns, region_map, window=12):
    """
    Region Alignment: correlación media entre los bloques geográficos.
    Alta correlación = mercados sincronizados.
    Baja correlación = rotación geográfica activa.
    """
    regions = list(region_map.keys())
    region_returns = pd.DataFrame()
    
    for region, assets in region_map.items():
        cols = [a for a in assets if a in weekly_returns.columns]
        if cols:
            region_returns[region] = weekly_returns[cols].mean(axis=1)
    
    if len(region_returns.columns) < 2:
        return 0.5  # neutral, sin señal
    
    # Correlación media entre regiones (ventana móvil)
    corr_matrix = region_returns.rolling(window).corr().dropna()
    if corr_matrix.empty:
        return 0.5
    
    # Última matriz de correlación
    last_corr = corr_matrix.iloc[-len(regions):]
    mean_corr = last_corr.values[np.triu_indices_from(last_corr.values, k=1)].mean()
    
    return mean_corr if pd.notna(mean_corr) else 0.5

def compute_coherence_v4(flow_pressure_spy, flow_pressure_global, risk_direction_spy, risk_direction_global):
    """
    Coherence v4: combina alineación de flujo y dirección en un solo score.
    Usa correlación de Pearson de las últimas 12 semanas.
    Rango [-1, +1].
    """
    window = 12
    if len(flow_pressure_spy) < window:
        return 0.0
    
    # Correlación de flow pressure
    flow_corr = flow_pressure_spy.iloc[-window:].corr(flow_pressure_global.iloc[-window:])
    # Correlación de risk direction
    risk_corr = risk_direction_spy.iloc[-window:].corr(risk_direction_global.iloc[-window:])
    
    if pd.isna(flow_corr) or pd.isna(risk_corr):
        return 0.0
    
    # Media de ambas correlaciones
    coherence = (flow_corr + risk_corr) / 2
    return np.clip(coherence, -1, 1)

def compute_dominance_flag(flow_spy, flow_others_median, threshold=2.0):
    """
    Detecta si SPY está dominando desproporcionadamente la señal de flujo.
    Retorna True si |flow_SPY| > 2 * mediana(|flow_ezu|, |flow_ewj|, |flow_eem|).
    """
    return abs(flow_spy) > threshold * abs(flow_others_median)