"""Contrato FUTURE_SETTLEMENT (FU-021-5 Fase 2.2 + FU-021-3C-bis).

Tras FU-021-3C-bis: universo reducido a BZ=F y CL=F. GC=F, HG=F y NG=F
migrados a SPOT_COMMODITY (spot, no futuros).

Semantica temporal: effective_date = settlement_date del exchange
(tipicamente D-1 laborable). expected = ultimo dia laborable.
max_lag=1, min_coverage=1.0.

Fuente: OilPriceAPI /v1/futures/ice-brent, /v1/futures/ice-wti.
Campo consumido: 'close' (proxy del settlement oficial ICE/NYMEX,
diferencia tipica <0.5%). NO es settlement oficial VWAP.
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


class FutureSettlement(TemporalContract):
    name = "FUTURE_SETTLEMENT"
    family = "FUTURE"
    session_calendar = "CME|ICE"
    max_lag_days = 1
    min_coverage = 1.0
    eligible_universe = ["BZ=F", "CL=F"]
    per_ticker_lag = {"BZ=F": 1, "CL=F": 1}
    per_pair_max_lag = None
    activation_req = None

    settlement_semantics = "close_proxy"

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
