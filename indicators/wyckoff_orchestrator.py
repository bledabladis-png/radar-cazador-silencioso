"""Orquestador de fases Wyckoff por stock.

Corre las cascadas de acumulacion y distribucion independientemente y
arbitra para emitir UNA SOLA fase por stock y fecha. Requisito de diseno:
nunca debe haber dos senales contradictorias peleando por la razon.

Tabla de decision (aprobada):
  dist CONFIRMED + acc CONFIRMED            -> CONFLICT
  dist CONFIRMED + acc != CONFIRMED         -> DISTRIBUTION
  dist != CONFIRMED + acc CONFIRMED         -> ACCUMULATION
  dist NOT_EVALUABLE + acc NOT_EVALUABLE    -> INSUFFICIENT_DATA (si falta
                                                de datos real; si es "no BC
                                                / no SC", RANGE)
  resto                                     -> RANGE

No se emiten MARKUP ni MARKDOWN (fuera de alcance v1: sin derivacion
solida y consistente, no se inventan).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from indicators.wyckoff_event import ClassifiedBreak
from indicators.wyckoff_distribution import classify_distribution_at
from indicators.wyckoff_accumulation import classify_accumulation_at


WyckoffPhase = Literal[
    "ACCUMULATION", "DISTRIBUTION", "RANGE", "CONFLICT", "INSUFFICIENT_DATA",
]


# Reasons de NOT_EVALUABLE que indican falta de datos real (no "no BC/SC").
_INSUFFICIENT_DATA_REASONS = frozenset({
    "empty_df_up_to_as_of",
    "nan_in_features",
    "insufficient_data",
})


@dataclass(frozen=True)
class FinalPhase:
    ticker: str
    as_of: pd.Timestamp
    phase: WyckoffPhase
    dist_classification: str
    acc_classification: str
    dist_evidence: dict[str, Any] = field(default_factory=dict)
    acc_evidence: dict[str, Any] = field(default_factory=dict)
    conflict_reason: str | None = None


def _is_data_insufficient(r: ClassifiedBreak) -> bool:
    """True si el ClassifiedBreak es NOT_EVALUABLE por falta de datos reales
    (no por 'no hay BC/SC en la ventana')."""
    if r.classification != "NOT_EVALUABLE":
        return False
    reason = r.evidence.get("reason", "")
    if reason in _INSUFFICIENT_DATA_REASONS:
        return True
    if isinstance(reason, str) and reason.startswith("score_error"):
        return True
    return False


def _decide(dist: ClassifiedBreak, acc: ClassifiedBreak) -> tuple[WyckoffPhase, str | None]:
    """Aplica la tabla de decision. Devuelve (phase, conflict_reason)."""
    dist_conf = dist.classification == "DISTRIBUTION_CONFIRMED"
    acc_conf = acc.classification == "ACCUMULATION_CONFIRMED"

    if dist_conf and acc_conf:
        return "CONFLICT", (
            f"dist={dist.classification} y acc={acc.classification} "
            f"confirmados simultaneamente en {dist.as_of.date()}"
        )
    if dist_conf:
        return "DISTRIBUTION", None
    if acc_conf:
        return "ACCUMULATION", None

    dist_ins = _is_data_insufficient(dist)
    acc_ins = _is_data_insufficient(acc)
    if dist_ins and acc_ins:
        return "INSUFFICIENT_DATA", None

    return "RANGE", None


def classify_phase_at(df, ticker, as_of, *,
                      dist_kwargs=None, acc_kwargs=None) -> FinalPhase:
    """Ejecuta ambas cascadas hasta as_of y emite la fase final."""
    dist = classify_distribution_at(df, ticker, as_of, **(dist_kwargs or {}))
    acc = classify_accumulation_at(df, ticker, as_of, **(acc_kwargs or {}))

    phase, conflict = _decide(dist, acc)
    return FinalPhase(
        ticker=ticker,
        as_of=as_of,
        phase=phase,
        dist_classification=dist.classification,
        acc_classification=acc.classification,
        dist_evidence=dist.evidence,
        acc_evidence=acc.evidence,
        conflict_reason=conflict,
    )


def classify_phase(df, ticker, *, dist_kwargs=None, acc_kwargs=None) -> FinalPhase:
    """Wrapper con as_of = ultima barra del df."""
    as_of = df.index[-1]
    return classify_phase_at(df, ticker, as_of,
                             dist_kwargs=dist_kwargs, acc_kwargs=acc_kwargs)