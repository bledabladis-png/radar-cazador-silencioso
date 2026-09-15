"""Contrato RATE_YIELD (FU-021-5 Fase 2.2)."""
from __future__ import annotations

from src.temporal_contracts.index_eod import _IndexEODBase
from src.temporal_contracts._common import nyse_expected


class RateYield(_IndexEODBase):
    name = "RATE_YIELD"
    family = "RATE_YIELD"
    session_calendar = "NYSE"
    max_lag_days = 1
    min_coverage = 1.0
    eligible_universe = ["^FVX", "^TNX"]
    per_ticker_lag = {"^FVX": 1, "^TNX": 1}

    def _expected(self, reference_date):
        return nyse_expected(reference_date)