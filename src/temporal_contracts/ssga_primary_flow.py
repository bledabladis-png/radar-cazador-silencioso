# Contrato SSGA_PRIMARY_FLOW (auditoria 11-03, hallazgo 10.1).
#
# effective_date = max(Date) de outputs/history/etf_primary_flow.csv.
# expected_date  = ultimo cierre de mercado (last_expected_market_date).
# max_lag_days   = 5 (una semana de tolerancia).
# min_coverage   = 0.9.
#
# SSGA publica primary flow diario con lag 1-2 dias respecto al cierre.
# En produccion normal el status sera STALE (lag 1-2, no error).
# INSUFFICIENT solo si el lag supera 5 dias. PENDING si no hay CSV.
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.temporal_contracts.base import (
    STATUS_PENDING,
    TemporalContract,
    TemporalResolution,
    compute_status,
)


_REF = Path(__file__).resolve().parents[2]


def _default_history_path() -> Path:
    return _REF / "outputs" / "history" / "etf_primary_flow.csv"


def _to_date(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return pd.Timestamp(v).date()
    except (TypeError, ValueError):
        return None


def _expected_close(reference_date):
    if reference_date is None:
        return None
    try:
        from src.market_calendar import last_expected_market_date
        return _to_date(last_expected_market_date(pd.Timestamp(reference_date)))
    except (ImportError, TypeError, ValueError):
        return None


class SsgaPrimaryFlow(TemporalContract):
    name = "SSGA_PRIMARY_FLOW"
    family = "SSGA"
    session_calendar = "NYSE"
    max_lag_days = 5
    min_coverage = 0.9
    eligible_universe = []
    per_ticker_lag = None
    per_pair_max_lag = None
    activation_req = None

    def __init__(self, history_path=None):
        self.history_path = Path(history_path) if history_path is not None else _default_history_path()
        self.eligible_universe = []

    def _pending(self, expected):
        return TemporalResolution(
            contract_name=self.name,
            effective_date=None,
            expected_date=expected,
            lag_days=None,
            coverage=0.0,
            status=STATUS_PENDING,
        )

    def resolve(self, df_market, reference_date):
        expected = _expected_close(reference_date)
        if not self.history_path.exists():
            return self._pending(expected)
        try:
            df = pd.read_csv(self.history_path, usecols=["Date"])
        except (OSError, ValueError, KeyError, TypeError):
            return self._pending(expected)
        if df is None or df.empty:
            return self._pending(expected)
        effective = _to_date(df["Date"].max())
        if effective is not None and expected is not None:
            lag = (expected - effective).days
        else:
            lag = None
        coverage = 1.0
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
