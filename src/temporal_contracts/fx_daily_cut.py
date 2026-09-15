"""Contrato FX_DAILY_CUT (FU-021-5 Fase 2.2).

Cutoff 17:00 ET. Lag heterogeneo por par (per_pair_max_lag).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.temporal_contracts._common import (
    fx_expected,
    resolve_universe,
    to_date,
)
from src.temporal_contracts.base import (
    TemporalContract,
    TemporalResolution,
    compute_status,
)


class FxDailyCut(TemporalContract):
    name = "FX_DAILY_CUT"
    family = "FX"
    session_calendar = "FX_24h"
    max_lag_days = {"EURUSD=X": 0, "USDJPY=X": 0, "USDCNY=X": 1}
    min_coverage = 0.5
    eligible_universe = ["EURUSD=X", "USDCNY=X", "USDJPY=X"]
    per_ticker_lag = None
    per_pair_max_lag = {"EURUSD=X": 0, "USDJPY=X": 0, "USDCNY=X": 1}
    activation_req = None

    def resolve(
        self, df_market: pd.DataFrame, reference_date
    ) -> TemporalResolution:
        res = resolve_universe(
            df_market, self.eligible_universe, self.min_coverage
        )
        effective = to_date(res.get("date"))
        expected = fx_expected(reference_date)
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
            per_member=self._per_member(res),
        )

    @staticmethod
    def _per_member(res: dict) -> Optional[dict]:
        # Placeholder para futura descomposicion por par.
        return None