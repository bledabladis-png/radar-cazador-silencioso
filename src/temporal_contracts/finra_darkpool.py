# Contrato FINRA_DARKPOOL (auditoria 11-03, hallazgo 10.1).
#
# effective_date = max(week) de outputs/history/darkpool_history.csv.
# expected_date  = lunes mas reciente con deadline FINRA (14d).
# max_lag_days   = 45 (frontera STALE de FRESHNESS_FINRA = (20, 28, 45)).
# min_coverage   = 0.9.
#
# Semantica: FINRA publica con lag regulatorio estructural 2-4 semanas.
# En produccion normal el status sera STALE (no es un error, es el
# contrato). INSUFFICIENT solo si el lag supera 45 dias (anomalia).
# PENDING si el CSV no existe o esta vacio.
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.temporal_contracts.base import (
    STATUS_PENDING,
    TemporalContract,
    TemporalResolution,
    compute_status,
)


_REF = Path(__file__).resolve().parents[2]
FINRA_LAG_MIN_DAYS = 14


def _default_history_path() -> Path:
    return _REF / "outputs" / "history" / "darkpool_history.csv"


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


def _most_recent_monday(d):
    return d - timedelta(days=d.weekday())


def _expected_week(reference_date):
    if reference_date is None:
        return None
    try:
        ref = pd.Timestamp(reference_date).date()
    except (TypeError, ValueError):
        return None
    return _most_recent_monday(ref - timedelta(days=FINRA_LAG_MIN_DAYS))


class FinraDarkpool(TemporalContract):
    name = "FINRA_DARKPOOL"
    family = "FINRA"
    session_calendar = "FINRA"
    max_lag_days = 45
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
        expected = _expected_week(reference_date)
        if not self.history_path.exists():
            return self._pending(expected)
        try:
            df = pd.read_csv(self.history_path, usecols=["week"])
        except (OSError, ValueError, KeyError, TypeError):
            return self._pending(expected)
        if df is None or df.empty:
            return self._pending(expected)
        effective = _to_date(df["week"].max())
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
