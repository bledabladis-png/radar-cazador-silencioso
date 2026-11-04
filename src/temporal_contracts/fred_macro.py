# Contrato FRED_MACRO (auditoria 11-03, hallazgo 10.1).
#
# effective_date = date de outputs/state/liquidity_state.json.
# expected_date  = ultimo cierre de mercado (last_expected_market_date).
# max_lag_days   = 90 (= FRESHNESS_FRED[2], frontera STALE de la
#                  constante que ya usa src/report/freshness.py).
# min_coverage   = 0.9.
#
# Semantica: FRED tiene lag estructural (WALCL semanal, series
# mensuales). En produccion normal el status sera STALE (lag 1-30),
# NO es error. INSUFFICIENT solo si lag > 90 (degradacion real de
# la fuente). OK solo si effective == expected (raro en FRED).
#
# Nota: el contrato mide integridad temporal del state de liquidez,
# no la frescura de cada serie FRED por separado. La distincion por
# serie (WALCL/RRP/TGA/DGS10) queda como diseno futuro si se
# necesita. Hoy la fuente FRED se materializa en un unico state.
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.temporal_contracts.base import (
    STATUS_PENDING,
    TemporalContract,
    TemporalResolution,
    compute_status,
)


_REF = Path(__file__).resolve().parents[2]


def _default_state_path() -> Path:
    return _REF / "outputs" / "state" / "liquidity_state.json"


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


class FredMacro(TemporalContract):
    name = "FRED_MACRO"
    family = "FRED"
    session_calendar = "FRED"
    max_lag_days = 90
    min_coverage = 0.9
    eligible_universe = []
    per_ticker_lag = None
    per_pair_max_lag = None
    activation_req = None

    def __init__(self, state_path=None):
        self.state_path = Path(state_path) if state_path is not None else _default_state_path()
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
        if not self.state_path.exists():
            return self._pending(expected)
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError, TypeError):
            return self._pending(expected)
        if not isinstance(data, dict):
            return self._pending(expected)
        effective = _to_date(data.get("date"))
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
