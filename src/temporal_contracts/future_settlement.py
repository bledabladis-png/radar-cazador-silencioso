"""Contrato FUTURE_SETTLEMENT (FU-021-5 Fase 2.2).

Estado permanente: BLOCKED (dictamen Q-B.4).
No resuelve contra df_market. No lanza excepcion. Devuelve un
TemporalResolution con status=BLOCKED y effective_date=None.
"""
from __future__ import annotations

import pandas as pd

from src.temporal_contracts.base import (
    TemporalContract,
    TemporalResolution,
    STATUS_BLOCKED,
)


class FutureSettlement(TemporalContract):
    name = "FUTURE_SETTLEMENT"
    family = "FUTURE"
    session_calendar = "CME|ICE"
    max_lag_days = None
    min_coverage = 0.0
    eligible_universe = ["BZ=F", "CL=F", "GC=F", "HG=F", "NG=F"]
    per_ticker_lag = None
    per_pair_max_lag = None
    activation_req = {
        "provider_specific_contract": True,
        "official_settlement": True,
        "stable_identifier": True,
        "declared_rollover": True,
    }

    def resolve(
        self, df_market: pd.DataFrame, reference_date
    ) -> TemporalResolution:
        return TemporalResolution(
            contract_name=self.name,
            effective_date=None,
            expected_date=None,
            lag_days=None,
            coverage=None,
            status=STATUS_BLOCKED,
        )