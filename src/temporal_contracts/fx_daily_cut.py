"""Contrato FX_DAILY_CUT (FU-021-5 Fase 2.2).

Cutoff 17:00 ET. Lag heterogeneo por par (per_pair_max_lag).
"""
from __future__ import annotations

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
    STATUS_BLOCKED,
    STATUS_INSUFFICIENT,
    STATUS_OK,
    STATUS_PENDING,
    STATUS_STALE,
)


class FxDailyCut(TemporalContract):
    name = "FX_DAILY_CUT"
    family = "FX"
    session_calendar = "FX_24h"
    # FX-CUTOFF-LAG-1 (2026-11-12): max_lag=1 para EURUSD/JPY (era 0).
    # Motivo: el close FX se publica ~17:00 ET. Los runs scheduled
    # corren mayoritariamente ANTES del cutoff del dia. A esa hora el
    # ultimo close disponible es D-1. Lag=1 es lo esperado por cutoff,
    # no INSUFFICIENT. Con max=0 todos los slots pre-cutoff marcaban
    # INSUFFICIENT, propagando WARN en check_temporal_contracts_result.
    max_lag_days = {"EURUSD=X": 1, "USDJPY=X": 1, "USDCNY=X": 1}
    min_coverage = 0.5
    eligible_universe = ["EURUSD=X", "USDCNY=X", "USDJPY=X"]
    per_ticker_lag = None
    per_pair_max_lag = {"EURUSD=X": 1, "USDJPY=X": 1, "USDCNY=X": 1}
    activation_req = None

    def resolve(
        self, df_market: pd.DataFrame, reference_date
    ) -> TemporalResolution:
        """Resuelve per-par (FX-PER-PAIR 2026-10-31, Fase 2 B1).

        Antes colapsaba per_pair_max_lag a max() y todos los pares
        heredaban el lag mas permisivo. Bug: EURUSD con lag 1 (max 0)
        se clasificaba STALE en el agregado. Ahora cada par tiene su
        propio status, y el agregado es el peor (BLOCKED > INSUFFICIENT
        > STALE > OK). per_member devuelve el detalle por par.

        FX-CUTOFF-LAG-1 (2026-11-12): max_lag=1 para EURUSD/JPY. Ver
        comentario en la clase. Lag 1 = STALE (esperado pre-cutoff),
        no INSUFFICIENT.
        """
        expected = fx_expected(reference_date)
        per_member = {}
        _rank = {STATUS_BLOCKED: 4, STATUS_INSUFFICIENT: 3,
                 STATUS_STALE: 2, STATUS_PENDING: 1, STATUS_OK: 0}
        worst_status = STATUS_OK
        worst_effective = None
        worst_lag = None
        worst_coverage = None
        for pair in self.eligible_universe:
            res = resolve_universe(df_market, [pair], self.min_coverage)
            effective = to_date(res.get("date"))
            if effective is not None and expected is not None:
                lag = (expected - effective).days
            else:
                lag = res.get("lag_days")
            coverage = res.get("coverage")
            if isinstance(self.max_lag_days, dict):
                max_lag_pair = self.max_lag_days.get(pair)
            else:
                max_lag_pair = self.max_lag_days
            status_pair = compute_status(
                effective_date=effective,
                expected_date=expected,
                lag_days=lag,
                coverage=coverage,
                max_lag_days=max_lag_pair,
                min_coverage=self.min_coverage,
            )
            per_member[pair] = {
                "effective_date": effective,
                "lag_days": lag,
                "coverage": coverage,
                "max_lag_days": max_lag_pair,
                "status": status_pair,
            }
            if _rank.get(status_pair, 0) > _rank.get(worst_status, 0):
                worst_status = status_pair
                worst_effective = effective
                worst_lag = lag
                worst_coverage = coverage
            elif worst_effective is None and effective is not None:
                worst_effective = effective
                worst_lag = lag
                worst_coverage = coverage
        return TemporalResolution(
            contract_name=self.name,
            effective_date=worst_effective,
            expected_date=expected,
            lag_days=worst_lag,
            coverage=worst_coverage,
            status=worst_status,
            per_member=per_member,
        )
