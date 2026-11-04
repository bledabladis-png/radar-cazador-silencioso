# Contrato SEC_13F (auditoria 11-03, hallazgo 10.1).
#
# effective_date = max(PERIODOFREPORT) del quarter mas reciente presente.
# expected_date  = quarter-end mas reciente cuyo 13F ya deberia estar
#                  filed segun reference_date. 13F vence 45 dias despues
#                  del period_end (regla SEC).
# max_lag_days   = 90 (permite un quarter completo de lag antes de
#                  INSUFFICIENT).
# min_coverage   = 0.9. coverage = presencia del fichero SUBMISSION.parquet
#                  del quarter mas reciente (1.0 presente, 0.0 ausente).
#
# Resuelve leyendo data/sec_13f/processed/ directamente. Ignora df_market.
# Coherente con el contrato del item: "cada fuente declara as_of_date
# propia"; la fuente aqui es el dataset SEC 13F en disco, no un DataFrame.
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
SEC_13F_DEADLINE_DAYS = 45


def _default_data_dir() -> Path:
    return _REF / "data" / "sec_13f" / "processed"


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
    """Quarter-end mas reciente cuyo 13F ya deberia estar filed."""
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
    while q_end + timedelta(days=SEC_13F_DEADLINE_DAYS) > ref:
        q_end = _prev_quarter_end(q_end)
        safety += 1
        if safety > 40:
            return None
    return q_end


class Sec13F(TemporalContract):
    name = "SEC_13F"
    family = "SEC"
    session_calendar = "SEC"
    max_lag_days = 90
    min_coverage = 0.9
    eligible_universe = []
    per_ticker_lag = None
    per_pair_max_lag = None
    activation_req = None

    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir is not None else _default_data_dir()
        self.eligible_universe = []

    def _latest_quarter_dir(self):
        if not self.data_dir.exists():
            return None
        quarters = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if not quarters:
            return None
        return quarters[-1]

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
        qdir = self._latest_quarter_dir()
        if qdir is None:
            return self._pending(expected)
        sub_path = qdir / "SUBMISSION.parquet"
        if not sub_path.exists():
            return self._pending(expected)
        try:
            df = pd.read_parquet(sub_path, columns=["PERIODOFREPORT"])
        except (OSError, ValueError, KeyError, TypeError):
            return self._pending(expected)
        if df is None or df.empty:
            return self._pending(expected)
        effective = _to_date(df["PERIODOFREPORT"].max())
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
