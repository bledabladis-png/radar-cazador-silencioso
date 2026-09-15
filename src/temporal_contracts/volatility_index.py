"""Contrato VOLATILITY_INDEX (FU-021-5 Fase 2.2).

Hereda de INDEX_EOD_USA (mismo calendario CBOE/NYSE, cierre 16:15 ET).
"""
from __future__ import annotations

from src.temporal_contracts.index_eod import _IndexEODBase
from src.temporal_contracts._common import nyse_expected


class VolatilityIndex(_IndexEODBase):
    name = "VOLATILITY_INDEX"
    family = "INDEX"
    session_calendar = "NYSE"
    max_lag_days = 1
    min_coverage = 1.0
    eligible_universe = ["^VIX", "^VIX3M", "^VXN"]
    per_ticker_lag = {"^VIX": 1, "^VIX3M": 1, "^VXN": 1}

    def _expected(self, reference_date):
        return nyse_expected(reference_date)