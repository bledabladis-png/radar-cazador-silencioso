# -*- coding: utf-8 -*-
"""
Volatilidad estructural v1.0
Describe régimen de volatilidad implícita, term structure y posicionamiento en opciones.
No alimenta motores, scores, pesos ni State Machine.
No incluye Dark Pool.
"""
import pandas as pd
import numpy as np
from src.utils import get_col

def compute_volatility_structure(df_market, pcr_data=None, vix_ticker='^VIX', vix3m_ticker='^VIX3M'):
    """
    Calcula métricas de volatilidad estructural.
    pcr_data: dict con 'zscore' y 'percentile_20d' o None.
    Devuelve DataFrame con una fila.
    """
    def _get_series(df, ticker):
        try:
            return get_col(df, ticker, 'Close')
        except (KeyError, TypeError):
            if ticker in df.columns:
                return df[ticker].astype(float)
            return pd.Series(dtype=float)

    vix = _get_series(df_market, vix_ticker).dropna()
    if len(vix) < 15:
        return pd.DataFrame()

    vix_level = vix.iloc[-1]

    # Percentiles
    if len(vix) >= 15:
        vix_p20 = vix.rolling(20, min_periods=15).rank(pct=True).iloc[-1]
    else:
        vix_p20 = np.nan
    if len(vix) >= 45:
        vix_p60 = vix.rolling(60, min_periods=45).rank(pct=True).iloc[-1]
    else:
        vix_p60 = np.nan

    # Term structure
    try:
        vix3m = _get_series(df_market, vix3m_ticker).dropna()
        if len(vix3m) >= 1:
            vix3m_level = vix3m.iloc[-1]
            term_ratio = vix3m_level / vix_level
            term_slope = vix3m_level - vix_level
            if term_ratio > 1.05:
                term_reading = 'Contango (vol. implícita mayor horizonte superior a inmediata)'
            elif term_ratio < 0.95:
                term_reading = 'Backwardation (vol. implícita inmediata superior a mayor horizonte)'
            else:
                term_reading = 'Curva plana'
        else:
            term_ratio = np.nan
            term_slope = np.nan
            term_reading = 'N/D'
    except KeyError:
        term_ratio = np.nan
        term_slope = np.nan
        term_reading = 'N/D'

    # PCR oficial
    if pcr_data is not None:
        pcr_z = pcr_data.get('zscore', np.nan)
        pcr_p20 = pcr_data.get('percentile_20d', np.nan)
    else:
        pcr_z = np.nan
        pcr_p20 = np.nan

    # Lectura de volatilidad según percentil 20d
    if pd.isna(vix_p20):
        vol_reading = 'N/D'
    elif vix_p20 >= 0.8:
        vol_reading = 'Volatilidad elevada reciente'
    elif vix_p20 >= 0.6:
        vol_reading = 'Volatilidad por encima de la media'
    elif vix_p20 >= 0.4:
        vol_reading = 'Volatilidad media'
    elif vix_p20 >= 0.2:
        vol_reading = 'Volatilidad por debajo de la media'
    else:
        vol_reading = 'Volatilidad reducida'

    return pd.DataFrame([{
        'date': pd.Timestamp.now().normalize(),
        'vix_level': vix_level,
        'vix_percentile_20d': vix_p20,
        'vix_percentile_60d': vix_p60,
        'term_structure_ratio': term_ratio,
        'term_structure_slope': term_slope,
        'pcr_zscore': pcr_z,
        'pcr_percentile_20d': pcr_p20,
        'volatility_reading': vol_reading,
        'term_structure_reading': term_reading,
        'n_valid_vix_20d': min(len(vix), 20),
        'n_valid_vix_60d': min(len(vix), 60),
        'n_valid_pcr_20d': np.nan if pd.isna(pcr_p20) else 20,
    }])