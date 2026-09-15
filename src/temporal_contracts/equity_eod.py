"""Contrato EQUITY_EOD (FU-021-5 Fase 2)."""
from __future__ import annotations

import pandas as pd

from src.temporal_contracts._common import (
    get_universe_by_class,
    nyse_expected,
    resolve_universe,
    to_date,
)
from src.temporal_contracts.base import (
    TemporalContract,
    TemporalResolution,
    compute_status,
)


class EquityEOD(TemporalContract):
    name = "EQUITY_EOD"
    family = "EQUITY"
    session_calendar = "NYSE"
    max_lag_days = 0
    min_coverage = 0.90
    per_ticker_lag = None
    per_pair_max_lag = None
    activation_req = None

    def __init__(self) -> None:
        self.eligible_universe: list = []

    def resolve(
        self, df_market: pd.DataFrame, reference_date
    ) -> TemporalResolution:
        universe = get_universe_by_class(df_market, "EQUITY")
        self.eligible_universe = universe
        res = resolve_universe(df_market, universe, self.min_coverage)

        effective = to_date(res.get("date"))
        expected = nyse_expected(reference_date)
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