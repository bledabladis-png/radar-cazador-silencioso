# -*- coding: utf-8 -*-
"""Contrato SEC_NPORT.

effective_date = max(REPORT_DATE) de
outputs/history/sec_nport_position_change_quarterly.csv (cierre del
trimestre). expected_date = quarter-end mas reciente cuyo N-PORT ya deberia
estar filed segun reference_date. N-PORT vence 60 dias despues del cierre
del trimestre (deadline SEC). max_lag_days = 180 (segundo umbral de
FRESHNESS_SEC: ritmo trimestral admite lag estructural de hasta ~5 meses
cuando el siguiente trimestre aun no ha cerrado). min_coverage = 0.9.

Semantica analoga a SEC_13F (deadline 45d) pero con 60d. Status normal:
OK cuando el trimestre mas reciente con deadline pasado coincide con el
del CSV. STALE si el CSV va un trimestre por detras. INSUFFICIENT si va
dos o mas.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.temporal_contracts.base import (
    STATUS_PENDING,
    TemporalContract,
    TemporalResolution,
    compute_status,
)


_REF = Path(__file__).resolve().parents[2]
SEC_NPORT_DEADLINE_DAYS = 60


def _default_history_path() -> Path:
    return _REF / "outputs" / "history" / "sec_nport_position_change_quarterly.csv"


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


def _prev_quarter_end(q):
    if q.month == 3:
        return date(q.year - 1, 12, 31)
    if q.month == 6:
        return date(q.year, 3, 31)
    if q.month == 9:
        return date(q.year, 6, 30)
    return date(q.year, 9, 30)


def _expected_quarter_end(reference_date):
    """Quarter-end mas reciente cuyo N-PORT ya deberia estar filed."""
    if reference_date is None:
        return None
    try:
        ref = pd.Timestamp(reference_date).date()
    except (TypeError, ValueError):
        return None
    y, m = ref.year, ref.month
    if m <= 3:
        q_end = date(y - 1, 12, 31)
    elif m <= 6:
        q_end = date(y, 3, 31)
    elif m <= 9:
        q_end = date(y, 6, 30)
    else:
        q_end = date(y, 9, 30)
    safety = 0
    while q_end + timedelta(days=SEC_NPORT_DEADLINE_DAYS) > ref:
        q_end = _prev_quarter_end(q_end)
        safety += 1
        if safety > 40:
            return None
    return q_end


class SecNport(TemporalContract):
    name = "SEC_NPORT"
    family = "SEC"
    session_calendar = "SEC"
    max_lag_days = 180
    min_coverage = 0.9
    eligible_universe = []
    per_ticker_lag = None
    per_pair_max_lag = None
    activation_req = None

    def __init__(self, history_path=None):
        self.history_path = (
            Path(history_path)
            if history_path is not None
            else _default_history_path()
        )
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
        expected = _expected_quarter_end(reference_date)
        if not self.history_path.exists():
            return self._pending(expected)
        try:
            df = pd.read_csv(self.history_path, usecols=["REPORT_DATE"])
        except (OSError, ValueError, KeyError, TypeError):
            return self._pending(expected)
        if df is None or df.empty:
            return self._pending(expected)
        effective = _to_date(df["REPORT_DATE"].max())
        if effective is None:
            return self._pending(expected)
        if expected is not None:
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
