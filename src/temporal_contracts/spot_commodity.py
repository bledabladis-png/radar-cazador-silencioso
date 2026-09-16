"""Contrato SPOT_COMMODITY (FU-021-3C-bis).

Universo: GC=F (Gold), HG=F (Copper), NG=F (Natural Gas).
Fuente: OilPriceAPI /v1/prices/latest (spot).
Semantica temporal: effective_date = ultima fecha con dato;
expected = ultimo dia laborable; max_lag=1.

NO es un contrato de futuros. El nombre del ticker conserva =F por
compatibilidad con consumidores existentes (decision FU-021-3C-bis).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.temporal_contracts._common import (
    resolve_universe,
    to_date,
    weekday_expected,
)
from src.temporal_contracts.base import (
    TemporalContract,
    TemporalResolution,
    compute_status,
)


class SpotCommodity(TemporalContract):
    name = "SPOT_COMMODITY"
    family = "SPOT"
    session_calendar = "COMEX|NYMEX"
    max_lag_days = 1
    min_coverage = 1.0
    eligible_universe = ["GC=F", "HG=F", "NG=F"]
    per_ticker_lag = {"GC=F": 1, "HG=F": 1, "NG=F": 1}
    per_pair_max_lag = None
    activation_req = None

    settlement_semantics = "spot_reference"

    def _expected(self, reference_date) -> Optional:
        return weekday_expected(reference_date)

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
