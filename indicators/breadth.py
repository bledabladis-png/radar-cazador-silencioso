# -*- coding: utf-8 -*-
"""
indicators/breadth.py -- Amplitud de Mercado Sectorial (v3.15 corregido)
Calcula el %% de sectores sobre EMAs y nuevos maximos/minimos de 52 semanas.
Usa breadth_core.py para NH/NL con shift(1).
"""
from dataclasses import dataclass

import numpy as np
from config.settings import BREADTH_EMA_FAST, BREADTH_EMA_MEDIUM, BREADTH_EMA_SLOW
import pandas as pd
from src.utils import get_col
from config.tickers import MARKET_TICKERS
from indicators.breadth_core import compute_new_highs_lows, validate_coverage

@dataclass(frozen=True)
class BreadthResult:
    """Resultado de compute_breadth con denominadores dinamicos.

    2026-11-21 (S39, 13.11): antes compute_breadth retornaba 5-tuple
    (b20, b50, b200, nh, nl) donde el denominador implicito era
    EXPECTED_SECTOR_COUNT (11). Si faltaban sectores, los porcentajes
    se subestimaban silenciosamente (X/11 con solo 8 sectores con datos).
    Ahora cada metrica lleva su propio denominador como Series.

    expected: total de sectores del universo (len(MARKET_TICKERS['sectors'])).
              Fuente unica de verdad. Coincide con EXPECTED_SECTOR_COUNT
              (verificado por centinela en tests).
    denom_*: Series con el numero de sectores con datos validos por fecha.
             Para b20/b50/b200: columnas con close y EMA no-NaN.
             Para nh/nl: tickers con >= window sesiones validas.
    """
    b20: pd.Series
    b50: pd.Series
    b200: pd.Series
    nh: pd.Series
    nl: pd.Series
    denom_ema20: pd.Series
    denom_ema50: pd.Series
    denom_ema200: pd.Series
    denom_nhnl: pd.Series
    expected: int


def compute_breadth(df):
    sectors = MARKET_TICKERS['sectors']
    expected = len(sectors)
    
    closes = pd.DataFrame(index=df.index)
    highs = pd.DataFrame(index=df.index)
    lows = pd.DataFrame(index=df.index)

    for s in sectors:
        try:
            closes[s] = get_col(df, s, 'Close')
            highs[s] = get_col(df, s, 'High')
            lows[s] = get_col(df, s, 'Low')
        except KeyError:
            closes[s] = np.nan
            highs[s] = np.nan
            lows[s] = np.nan

    # Validar cobertura
    validate_coverage(closes, expected, "Breadth Sectorial")

    # EMAs con adjust=False (estandar de analisis tecnico)
    ema20 = closes.ewm(span=BREADTH_EMA_FAST, adjust=False, min_periods=20).mean()
    ema50 = closes.ewm(span=BREADTH_EMA_MEDIUM, adjust=False, min_periods=50).mean()
    ema200 = closes.ewm(span=BREADTH_EMA_SLOW, adjust=False, min_periods=200).mean()

    # 2026-11-21 (S39, 13.11): denominadores dinamicos. Cada metrica
    # divide por el numero de sectores con datos validos en esa fecha,
    # no por expected. Si Y == expected (caso normal), el resultado es
    # identico al anterior. Si Y < expected (sector nuevo, fallo de
    # proveedor), el % es representativo de los sectores disponibles.
    valid_20 = closes.notna() & ema20.notna()
    valid_50 = closes.notna() & ema50.notna()
    valid_200 = closes.notna() & ema200.notna()

    _denom20 = valid_20.sum(axis=1).replace(0, np.nan)
    _denom50 = valid_50.sum(axis=1).replace(0, np.nan)
    _denom200 = valid_200.sum(axis=1).replace(0, np.nan)

    b20 = ((closes > ema20) & valid_20).sum(axis=1) / _denom20
    b50 = ((closes > ema50) & valid_50).sum(axis=1) / _denom50
    b200 = ((closes > ema200) & valid_200).sum(axis=1) / _denom200

    # NH/NL usando la funcion comun (shift(1) + High/Low)
    nh, nl, nhnl_valid = compute_new_highs_lows(highs)
    _denom_nhnl = nhnl_valid.replace(0, np.nan)
    nh_pct = nh / _denom_nhnl
    nl_pct = nl / _denom_nhnl

    return BreadthResult(
        b20=b20,
        b50=b50,
        b200=b200,
        nh=nh_pct,
        nl=nl_pct,
        denom_ema20=_denom20,
        denom_ema50=_denom50,
        denom_ema200=_denom200,
        denom_nhnl=_denom_nhnl,
        expected=expected,
    )

