"""Helpers compartidos para los contratos temporales (FU-021-5 Fase 2)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from src.effective_date import resolve_effective_date
from src.instrument_registry import get_instrument_class
from src.market_calendar import last_expected_market_date


def extract_close(df_market: pd.DataFrame) -> pd.DataFrame:
    """Extrae el sub-DataFrame de Close desde df_market.

    Si df_market tiene columnas MultiIndex (field, ticker), devuelve
    solo las columnas Close con tickers planos. Si ya tiene columnas
    planas, devuelve df_market tal cual.
    """
    if df_market is None or df_market.empty:
        return df_market
    if isinstance(df_market.columns, pd.MultiIndex):
        return df_market.xs("Close", axis=1, level=0)
    return df_market


def get_universe_by_class(df_market: pd.DataFrame, klass: str) -> list:
    """Tickers de df_market cuya clase economica es klass.

    Preserva el orden de aparicion en las columnas.
    """
    close = extract_close(df_market)
    if close is None or close.empty:
        return []
    return [t for t in close.columns if get_instrument_class(t) == klass]


def to_date(value) -> Optional[date]:
    """Normaliza Timestamp/datetime/date a date (None si no aplicable)."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def nyse_expected(reference_date) -> Optional[date]:
    """Ultima sesion NYSE esperada en reference_date."""
    try:
        return to_date(last_expected_market_date(reference_date))
    except Exception:
        return None


def weekday_expected(reference_date) -> Optional[date]:
    """Ultima sesion L-V en reference_date. PROVISIONAL.

    No usa calendario de festivos locales. Se sustituira por
    calendarios oficiales de LSE/XETRA/BME/EURONEXT en fase posterior.
    """
    d = to_date(reference_date)
    if d is None:
        return None
    while d.weekday() >= 5:
        d = d - timedelta(days=1)
    return d


def fx_expected(reference_date) -> Optional[date]:
    """Fecha esperada FX: fecha de reference_date (cutoff 17:00 ET)."""
    return to_date(reference_date)


def resolve_universe(
    df_market: pd.DataFrame,
    universe: list,
    min_coverage: float,
) -> dict:
    """Envoltorio delgado sobre resolve_effective_date (FU-020)."""
    return resolve_effective_date(df_market, universe, min_coverage=min_coverage)