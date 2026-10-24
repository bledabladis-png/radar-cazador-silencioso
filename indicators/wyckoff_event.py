"""Tipos y dataclasses comunes para las cascadas Wyckoff.

Este modulo NO contiene detectores. Solo contratos:
- WyckoffEvent: evento atomico (BC, AR, SOW, SC, Spring, SOS, ...)
- Range: estructura de rango con soporte/resistencia
- ClassifiedBreak: salida final de una cascada (distribucion o acumulacion)

Los detectores y cascadas viven en:
- wyckoff_distribution.py (BC -> AR -> Range -> SOW)
- wyckoff_accumulation.py (SC -> AR -> Range -> Spring/SOS)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


EventStatus = Literal[
    "CONFIRMED", "CANDIDATE", "NOT_OBSERVED", "NOT_EVALUABLE",
    "INVALIDATED", "REJECTED", "EXPIRED",
]

RangeState = Literal[
    "RANGE_FORMING", "RANGE_ACTIVE", "RANGE_BROKEN",
    "RANGE_INVALIDATED", "RANGE_EXPIRED", "NOT_EVALUABLE",
]

RangeType = Literal[
    "DISTRIBUTION_CANDIDATE", "REDISTRIBUTION_CANDIDATE",
    "ACCUMULATION_CANDIDATE", "UNCLASSIFIED",
]

FormationPath = Literal["BC_AR", "REPEATED_TESTS"]

# Clasificacion de distribucion (se mantiene el nombre por compatibilidad
# con el pipeline actual; en acumulacion se reutiliza el mismo Literal para
# evitar proliferacion de tipos, aunque los valores DISTRIBUTION/REDISTRIBUTION
# solo se emiten desde la cascada de distribucion).
Classification = Literal[
    "DISTRIBUTION_CONFIRMED",
    "REDISTRIBUTION_CONFIRMED",
    "BREAK_WITHOUT_DISTRIBUTION_CONTEXT",
    "BREAK_PENDING",
    "RANGE_ACTIVE_NO_BREAK",
    "NOT_EVALUABLE",
    # Valores exclusivos de la cascada de acumulacion:
    "ACCUMULATION_CONFIRMED",
    "BREAKOUT_PENDING",
    "SPRING_PENDING",
    "BREAKOUT_WITHOUT_ACCUMULATION_CONTEXT",
    "UTAD_PENDING",
    "DISTRIBUTION_FORMING",
    "ACCUMULATION_FORMING",
]


@dataclass(frozen=True)
class WyckoffEvent:
    ticker: str
    event_type: str
    event_date: pd.Timestamp | None
    available_at: pd.Timestamp | None
    status: EventStatus
    range_id: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Range:
    range_id: str
    ticker: str
    timeframe: str
    formation_path: FormationPath
    range_type: RangeType
    candidate_start_date: pd.Timestamp
    start_available_at: pd.Timestamp
    range_active_at: pd.Timestamp | None
    support: float
    resistance: float
    tolerance_frac: float
    tol_abs: float
    state: RangeState
    end: pd.Timestamp | None = None
    closed_at: pd.Timestamp | None = None
    break_direction: Literal["UP", "DOWN"] | None = None
    trend_context: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassifiedBreak:
    ticker: str
    as_of: pd.Timestamp
    range_id: str | None
    classification: Classification
    structural_state: str
    break_direction: Literal["UP", "DOWN"] | None
    range_type: RangeType | None
    trend_direction: str | None
    sow_status: str | None
    coherence_error: str | None
    evidence: dict[str, Any] = field(default_factory=dict)
    range_active_at: pd.Timestamp | None = None
    resolution_event_date: pd.Timestamp | None = None
    range_support: float | None = None
    range_resistance: float | None = None
    range_tol_abs: float | None = None