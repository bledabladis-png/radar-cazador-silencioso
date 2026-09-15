"""Base infrastructure for FU-021-5 temporal contracts (Fase 1).

Reference: docs/auditoria/FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL.md
           docs/auditoria/DICTAMEN_FU-021-5_PARTE_B.md
           docs/auditoria/DICTAMEN_PLAN_IMPLEMENTACION.md
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Union

import pandas as pd


# --- FSM status constants (FU-021-5 A.10) ---

STATUS_PENDING = "PENDING"
STATUS_OK = "OK"
STATUS_STALE = "STALE"
STATUS_INSUFFICIENT = "INSUFFICIENT"
STATUS_BLOCKED = "BLOCKED"

VALID_STATUSES = (
    STATUS_PENDING,
    STATUS_OK,
    STATUS_STALE,
    STATUS_INSUFFICIENT,
    STATUS_BLOCKED,
)


def compute_status(
    effective_date: Optional[date],
    expected_date: Optional[date],
    lag_days: Optional[int],
    coverage: Optional[float],
    max_lag_days: Optional[Union[int, dict]],
    min_coverage: Optional[float],
    activation_req_resolved: bool = True,
) -> str:
    """FSM de contratos temporales (FU-021-5 A.10).

    Orden de evaluacion:
      1. BLOCKED      -> activation_requirement no resuelto.
      2. PENDING      -> sin resolucion (effective_date y expected_date None).
      3. INSUFFICIENT -> coverage < min_coverage o lag_days > max_lag.
      4. OK           -> effective_date == expected_date.
      5. STALE        -> 0 < lag_days <= max_lag.
    """
    if not activation_req_resolved:
        return STATUS_BLOCKED

    if effective_date is None and expected_date is None:
        return STATUS_PENDING

    if isinstance(max_lag_days, dict):
        ml = max(max_lag_days.values()) if max_lag_days else 0
    else:
        ml = max_lag_days if max_lag_days is not None else 0

    if coverage is not None and min_coverage is not None and coverage < min_coverage:
        return STATUS_INSUFFICIENT
    if lag_days is not None and lag_days > ml:
        return STATUS_INSUFFICIENT

    if (
        effective_date is not None
        and expected_date is not None
        and effective_date == expected_date
    ):
        return STATUS_OK

    if lag_days is not None and 0 < lag_days <= ml:
        return STATUS_STALE

    return STATUS_PENDING


@dataclass
class TemporalResolution:
    """Resultado de resolver un contrato temporal."""

    contract_name: str
    effective_date: Optional[date]
    expected_date: Optional[date]
    lag_days: Optional[int]
    coverage: Optional[float]
    status: str
    per_member: Optional[dict] = None


@dataclass
class MarketDataBundle:
    """Transporte explicito de df_market + temporal_meta (FU-021-5 Q-P.3).

    `temporal_meta` es la autoridad. `df.attrs` puede espejarlo pero no
    sustituirlo.
    """

    df_market: pd.DataFrame
    temporal_meta: dict = field(default_factory=dict)


class TemporalContract(ABC):
    """Interfaz comun de los 9 contratos temporales (FU-021-5 A.1)."""

    name: str
    family: str
    eligible_universe: list
    session_calendar: str
    max_lag_days: Optional[Union[int, dict]]
    min_coverage: float
    per_ticker_lag: Optional[dict] = None
    per_pair_max_lag: Optional[dict] = None
    activation_req: Optional[dict] = None

    @abstractmethod
    def resolve(
        self, df_market: pd.DataFrame, reference_date: datetime
    ) -> TemporalResolution:
        """Resuelve effective_date, expected_date, lag_days, coverage, status."""

    def is_eligible(self, ticker: str) -> bool:
        """True si el ticker pertenece al universo del contrato."""
        return ticker in self.eligible_universe