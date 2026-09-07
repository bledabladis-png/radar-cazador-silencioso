# -*- coding: utf-8 -*-
"""
Dispersión entre sectores v1.0
Mide heterogeneidad cross-sectional de retornos sectoriales 20d.
No alimenta motores, scores, pesos ni State Machine.
Consume retornos oficiales desde run.py (sector_price_rank).
"""
import pandas as pd
import numpy as np

def compute_sector_dispersion(price_rank_list):
    """
    price_rank_list: lista de tuplas (ticker, ret) con retornos 20d.
    Devuelve DataFrame con una fila con métricas de dispersión.
    """
    if not price_rank_list:
        return pd.DataFrame()

    df = pd.DataFrame(price_rank_list, columns=['ticker','ret']).dropna()
    n_total = 11
    n_valid = len(df)
    coverage = n_valid / n_total if n_total else 0.0

    if n_valid < 8:
        return pd.DataFrame([{
            'date': pd.Timestamp.now().normalize(),
            'n_total': n_total,
            'n_valid': n_valid,
            'coverage': coverage,
            'range_pp': np.nan,
            'std_pp': np.nan,
            'mean_ret': np.nan,
            'dispersion_reading': 'N/D',
            'heterogeneity_type': 'N/D',
        }])

    ret = df['ret'].astype(float) * 100  # Convertir decimal a puntos porcentuales
    range_pp = ret.max() - ret.min()
    std_pp = ret.std(ddof=0)
    mean_ret = ret.mean()

    # Clasificación descriptiva
    if std_pp < 2:
        dispersion_reading = 'Muy baja'
    elif std_pp < 4:
        dispersion_reading = 'Baja'
    elif std_pp < 6:
        dispersion_reading = 'Moderada'
    elif std_pp < 8:
        dispersion_reading = 'Alta'
    else:
        dispersion_reading = 'Muy alta'

    # Heterogeneidad
    if range_pp >= 5 and std_pp >= 2.5:
        heterogeneity_type = 'Heterogeneidad amplia'
    else:
        heterogeneity_type = 'Heterogeneidad contenida'

    return pd.DataFrame([{
        'date': pd.Timestamp.now().normalize(),
        'n_total': n_total,
        'n_valid': n_valid,
        'coverage': coverage,
        'range_pp': range_pp,
        'std_pp': std_pp,
        'mean_ret': mean_ret,
        'dispersion_reading': dispersion_reading,
        'heterogeneity_type': heterogeneity_type,
    }])