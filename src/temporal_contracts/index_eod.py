"""Contratos INDEX_EOD_* (FU-021-5 Fase 2)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.temporal_contracts._common import (
    nyse_expected,
    resolve_universe,
    to_date,
    weekday_expected,
)
from src.temporal_contracts.base import (
    TemporalContract,
    TemporalResolution,
    compute_status,
)


class _IndexEODBase(TemporalContract):
    family = "INDEX"
    min_coverage = 1.0
    per_ticker_lag = None
    per_pair_max_lag = None
    activation_req = None

    def _expected(self, reference_date) -> Optional:
        raise NotImplementedError

    def resolve(
        self, df_market: pd.DataFrame, reference_date
    ) -> TemporalResolution:
        res = resolve_universe(
            df_market, self.eligible_universe, self.min_coverage
        )
        effective = to_date(res.get("date"))
        expected = self._expected(reference_date)
        if effective is not None and expected is not None:
            lag = (expected - effective).days
        else:
            lag = res.get("lag_days")
        coverage = res.get("coverage")
        status = compute_status(
            effective_date=effective,
            expected_date=expected,
            lag_days=lag,
            coverage=coverage,
            max_lag_days=self.max_lag_days,
            min_coverage=self.min_coverage,
        )
        return TemporalResolution(
            contract_name=self.name,
            effective_date=effective,
            expected_date=expected,
            lag_days=lag,
            coverage=coverage,
            status=status,
        )


class IndexEODUSA(_IndexEODBase):
    name = "INDEX_EOD_USA"
    session_calendar = "NYSE"
    max_lag_days = 1
    eligible_universe = ["^GSPC", "^DJI", "^NDX", "^RUT"]
    per_ticker_lag = {"^GSPC": 1, "^DJI": 1, "^NDX": 1, "^RUT": 1}

    def _expected(self, reference_date):
        return nyse_expected(reference_date)


class IndexEODEuropa(_IndexEODBase):
    name = "INDEX_EOD_EUROPA"
    session_calendar = "LSE|XETRA|BME|EURONEXT"
    max_lag_days = 5
    eligible_universe = ["^FTSE", "^GDAXI", "^IBEX", "^STOXX50E"]
    per_ticker_lag = {"^FTSE": 1, "^GDAXI": 2, "^IBEX": 2, "^STOXX50E": 2}

    def _expected(self, reference_date):
        return weekday_expected(reference_date)


class IndexEODCommodity(_IndexEODBase):
    name = "INDEX_EOD_COMMODITY"
    session_calendar = "NYSE"
    max_lag_days = 1
    eligible_universe = ["^SPGSCI"]
    per_ticker_lag = {"^SPGSCI": 1}

    def _expected(self, reference_date):
        return nyse_expected(reference_date)


class IndexEODCurrency(_IndexEODBase):
    name = "INDEX_EOD_CURRENCY"
    session_calendar = "ICE"
    max_lag_days = 1
    eligible_universe = ["DX-Y.NYB"]
    per_ticker_lag = {"DX-Y.NYB": 1}

    def _expected(self, reference_date):
        return nyse_expected(reference_date)