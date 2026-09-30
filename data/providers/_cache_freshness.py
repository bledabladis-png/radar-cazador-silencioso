# -*- coding: utf-8 -*-
"""Helper comun de cache freshness para providers europeos.

Fix K (2026-09-30): Euronext y BME usaban `(reference_date - last).days
<= 1`. Con ref=30-sep 23:33 y cache=29-sep, consideran "fresco" y no
refrescan. Xetra ya usaba la logica FU-018 (si hoy es sesion del
mercado y cerro, la cache debe contener HOY). Este helper unifica la
regla en los 3 providers.

Verificado 2026-09-30: 13 Euronext + 19 BME quedan en 29-sep por el
bug. stock_prices.parquet cae a 89.8% cobertura, market_data.parquet
al 100%. Consecuencia en el reporte: las 12 secciones derivadas
publican datos del 29 mientras el header dice 30.

Contrato:
  - last_date: Timestamp de la ultima observacion en cache.
  - ticker: para resolver el mercado via get_market.
  - reference_date: fecha del run (tz-aware o naive).
  - True si la cache es fresca respecto a reference_date.
  - Si el mercado no se reconoce: fallback legacy (<=1 dia).
  - Si hoy es sesion del mercado y ya cerro: cache debe contener HOY.
  - Si no: dentro de 1 dia se considera fresco.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from src.instrument_registry import get_market
from src.market_hours import is_trading_session, is_session_closed


def european_cache_is_fresh(
    last_date: pd.Timestamp,
    ticker: str,
    reference_date: Optional[datetime] = None,
) -> bool:
    """True si la cache contiene la ultima sesion EOD esperada.

    Replica la logica FU-018 de Xetra (data/providers/xetra_provider.py).
    """
    last = pd.Timestamp(last_date)
    if reference_date is None:
        return (pd.Timestamp.now().normalize() - last).days <= 1

    market = get_market(ticker)
    if market == "UNKNOWN":
        return (pd.Timestamp(reference_date).normalize() - last).days <= 1

    ref_date = pd.Timestamp(reference_date).date()
    if is_trading_session(market, ref_date) and is_session_closed(
        market, ref_date, reference_date
    ):
        return last.date() >= ref_date

    return (ref_date - last.date()).days <= 1
