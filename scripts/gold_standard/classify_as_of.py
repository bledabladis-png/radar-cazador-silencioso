# -*- coding: utf-8 -*-
"""Wrapper temporal/as-of para clasificar fase en un punto L.

Especificacion (protocolo v7, seccion 4.7):
  - NO modifica indicators/wyckoff_v1.py.
  - Inyecta FROZEN_V19_PARAMS explicitamente. Nunca defaults productivos.
  - as_of obligatorio: la clasificacion es en L, no en el ultimo valor.
  - Trunca df.loc[:as_of] de forma redundante (defensa en profundidad:
    idempotente con el truncamiento interno del modulo).
  - Equivalencia bit-a-bit con classify_wyckoff_phase(df.loc[:L], tk,
    as_of=L, sow_params=FROZEN_V19_PARAMS).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import pandas as pd

from indicators.wyckoff_v1 import classify_wyckoff_phase


# Parametros congelados de la candidata v1.9.
FROZEN_V19_PARAMS = {
    "window": 60,
    "x_atr": 0.25,
    "y_vol": 1.10,
    "max_age_m": 10,
}


def classify_as_of(
    df_ticker: pd.DataFrame,
    ticker: str,
    as_of: pd.Timestamp,
    sow_params: dict | None = None,
) -> str:
    """Clasifica la fase Wyckoff en L = as_of.

    df_ticker: DataFrame OHLCV por ticker (flat, no MultiIndex) o MultiIndex.
    ticker: str.
    as_of: timestamp obligatorio. No se permite None.
    sow_params: si None, usa FROZEN_V19_PARAMS.

    Devuelve uno de: MARKUP | ACCUMULATION | RANGE | DISTRIBUTION |
                     MARKDOWN | INSUFFICIENT_DATA.
    """
    if as_of is None:
        raise ValueError(
            "classify_as_of requiere as_of explicito. "
            "Para clasificar en el ultimo valor disponible, usar "
            "classify_wyckoff_phase sin as_of."
        )
    params = dict(sow_params) if sow_params is not None else dict(FROZEN_V19_PARAMS)
    truncated = df_ticker.loc[:as_of]
    return classify_wyckoff_phase(
        truncated, ticker, as_of=as_of, sow_params=params,
    )


def classify_as_of_batch(
    df: pd.DataFrame,
    ticker: str,
    as_of_list: list[pd.Timestamp],
    sow_params: dict | None = None,
) -> dict[pd.Timestamp, str]:
    """Clasifica en multiples L para el mismo ticker.

    df: OHLCV completo (no pre-truncado).
    """
    if sow_params is None:
        sow_params = dict(FROZEN_V19_PARAMS)
    out = {}
    for L in as_of_list:
        out[L] = classify_as_of(df, ticker, L, sow_params=sow_params)
    return out