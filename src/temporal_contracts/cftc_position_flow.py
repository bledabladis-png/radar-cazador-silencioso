# -*- coding: utf-8 -*-
"""Contrato CFTC_POSITION_FLOW.

effective_date = max(date) de outputs/history/cftc_position_flow.csv (martes
del reporte TFF semanal). expected_date = martes mas reciente con margen
(CFTC_LAG_MIN_DAYS = 10 dias: 3d publicacion + 7d seguridad). max_lag_days
= 45 (primer umbral de FRESHNESS_CFTC; 45d ya es anomalia per comentario
en config/settings.py:49). min_coverage = 0.9.

Semantica: CFTC TFF publica los viernes con datos del martes. En produccion
normal el status sera OK (CSV actualizado) o STALE (lag 1-2 semanas). Si
effective >= expected (CSV por delante del margen), status OK con lag=0:
no es lag del proveedor, es que el CSV esta al dia. INSUFFICIENT solo si el
lag supera 45 dias. PENDING si el CSV no existe o esta vacio.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.temporal_contracts.base import (
    STATUS_OK,
    STATUS_PENDING,
    TemporalContract,
    TemporalResolution,
    compute_status,
)


_REF = Path(__file__).resolve().parents[2]
CFTC_LAG_MIN_DAYS = 10


def _default_history_path() -> Path:
    return _REF / "outputs" / "history" / "cftc_position_flow.csv"


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


def _most_recent_tuesday(d):
    """Martes <= d. d.weekday(): lunes=0, martes=1, ..., domingo=6."""
    return d - timedelta(days=(d.weekday() - 1) % 7)


def _expected_tuesday(reference_date):
    if reference_date is None:
        return None
    try:
        ref = pd.Timestamp(reference_date).date()
    except (TypeError, ValueError):
        return None
    return _most_recent_tuesday(ref - timedelta(days=CFTC_LAG_MIN_DAYS))


class CftcPositionFlow(TemporalContract):
    name = "CFTC_POSITION_FLOW"
    family = "CFTC"
    session_calendar = "CFTC"
    max_lag_days = 45
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
        expected = _expected_tuesday(reference_date)
        if not self.history_path.exists():
            return self._pending(expected)
        try:
            df = pd.read_csv(self.history_path, usecols=["date"])
        except (OSError, ValueError, KeyError, TypeError):
            return self._pending(expected)
        if df is None or df.empty:
            return self._pending(expected)
        effective = _to_date(df["date"].max())
        if effective is None:
            return self._pending(expected)
        # CSV por delante del margen: no es lag del proveedor, es que el
        # CSV esta al dia. Status OK explicito (compute_status devolveria
        # PENDING con lag negativo).
        if expected is not None and effective >= expected:
            return TemporalResolution(
                contract_name=self.name,
                effective_date=effective,
                expected_date=expected,
                lag_days=0,
                coverage=1.0,
                status=STATUS_OK,
            )
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
