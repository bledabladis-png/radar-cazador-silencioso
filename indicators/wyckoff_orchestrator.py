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


def _is_dist_current(cb: ClassifiedBreak, current_price: float | None) -> bool:
    """Distribucion vigente si el precio esta DENTRO del rango +- tolerancia.

    Un rango esta vigente mientras el precio no lo ha roto por NINGUN lado:
    - Rotura alcista (price > resistance + tol): la distribucion fracaso.
    - Rotura bajista (price < support - tol): la distribucion se consumio.
    En ambos casos, el rango ya no describe la situacion actual.

    Definicion simetrica de rango, no umbral estadistico."""
    if current_price is None:
        return False
    if cb.range_support is None or cb.range_resistance is None or cb.range_tol_abs is None:
        return False
    price = float(current_price)
    tol = float(cb.range_tol_abs)
    return (float(cb.range_support) - tol) <= price <= (float(cb.range_resistance) + tol)


def _is_acc_current(cb: ClassifiedBreak, current_price: float | None) -> bool:
    """Acumulacion vigente si el precio esta DENTRO del rango +- tolerancia.

    Simetrica a _is_dist_current. Simetria de definicion: un rango esta
    vigente mientras el precio no lo ha roto por ningun lado."""
    if current_price is None:
        return False
    if cb.range_support is None or cb.range_resistance is None or cb.range_tol_abs is None:
        return False
    price = float(current_price)
    tol = float(cb.range_tol_abs)
    return (float(cb.range_support) - tol) <= price <= (float(cb.range_resistance) + tol)


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
    phase_since: pd.Timestamp | None = None
    event_phase: WyckoffPhase | None = None
    current_phase_basis: str | None = None


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


def _decide_event_phase(dist: ClassifiedBreak, acc: ClassifiedBreak) -> tuple[WyckoffPhase, str | None]:
    """Fase de EVENTO (historica): sin regla de vigencia por precio.
    Refleja 'que paso', no 'que pasa ahora'."""
    dist_conf = dist.classification == "DISTRIBUTION_CONFIRMED"
    acc_conf = acc.classification == "ACCUMULATION_CONFIRMED"

    if dist_conf and acc_conf:
        # Ambos con structural RANGE_ACTIVE simultaneo -> conflicto real
        if dist.structural_state == "RANGE_ACTIVE" and acc.structural_state == "RANGE_ACTIVE":
            return "CONFLICT", (
                f"ambas cascadas con RANGE_ACTIVE en {dist.as_of.date()}"
            )
        # Episodios historicos distintos: gana el mas reciente
        d_res = dist.resolution_event_date
        a_res = acc.resolution_event_date
        if d_res is None and a_res is None:
            return "RANGE", None
        if d_res is None:
            return "ACCUMULATION", None
        if a_res is None:
            return "DISTRIBUTION", None
        if d_res > a_res:
            return "DISTRIBUTION", None
        if a_res > d_res:
            return "ACCUMULATION", None
        # Misma fecha de resolucion -> CONFLICT real (raro, pero posible)
        return "CONFLICT", (
            f"ambas resueltas en {d_res.date()} (misma sesion)"
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


def _dist_derived_phase(cb: ClassifiedBreak, price: float | None) -> str | None:
    """Fase derivada de una cascada de distribucion confirmada."""
    if cb.classification != "DISTRIBUTION_CONFIRMED":
        return None
    if price is None or cb.range_support is None or cb.range_resistance is None or cb.range_tol_abs is None:
        return None
    tol = float(cb.range_tol_abs)
    if float(price) > float(cb.range_resistance) + tol:
        return "RANGE"
    if float(price) < float(cb.range_support) - tol:
        return "MARKDOWN"
    return "DISTRIBUTION"


def _acc_derived_phase(cb: ClassifiedBreak, price: float | None) -> str | None:
    """Fase derivada de una cascada de acumulacion confirmada."""
    if cb.classification != "ACCUMULATION_CONFIRMED":
        return None
    if price is None or cb.range_support is None or cb.range_resistance is None or cb.range_tol_abs is None:
        return None
    tol = float(cb.range_tol_abs)
    if float(price) < float(cb.range_support) - tol:
        return "RANGE"
    if float(price) > float(cb.range_resistance) + tol:
        return "MARKUP"
    return "ACCUMULATION"


def _decide_current_phase(
    dist: ClassifiedBreak, acc: ClassifiedBreak, current_price: float | None
) -> tuple[WyckoffPhase, str | None, str]:
    """Fase ACTUAL derivada de cascadas confirmadas + posicion del precio."""
    d_phase = _dist_derived_phase(dist, current_price)
    a_phase = _acc_derived_phase(acc, current_price)

    if d_phase is None and a_phase is None:
        dist_ins = _is_data_insufficient(dist)
        acc_ins = _is_data_insufficient(acc)
        if dist_ins and acc_ins:
            return ("INSUFFICIENT_DATA", None, "ambas cascadas sin datos suficientes")
        return ("RANGE", None, "sin cascada activa")

    if a_phase is None:
        return (d_phase, None, _basis_for(d_phase, "dist", current_price, dist))
    if d_phase is None:
        return (a_phase, None, _basis_for(a_phase, "acc", current_price, acc))

    if d_phase == "DISTRIBUTION" and a_phase == "ACCUMULATION":
        return ("CONFLICT",
                f"dist y acc vigentes en {dist.as_of.date()}",
                "ambas dentro del rango")

    if d_phase == "RANGE" and a_phase == "RANGE":
        return ("RANGE", None, "ambas cascadas fallidas (precio fuera por lado opuesto)")

    if d_phase == "RANGE":
        return (a_phase, None, _basis_for(a_phase, "acc", current_price, acc) + " (dist fallida)")
    if a_phase == "RANGE":
        return (d_phase, None, _basis_for(d_phase, "dist", current_price, dist) + " (acc fallida)")

    d_res = dist.resolution_event_date
    a_res = acc.resolution_event_date
    if d_res is None and a_res is None:
        return ("CONFLICT", f"ambas sin fecha ({d_phase} vs {a_phase})", "empate sin fecha")
    if d_res is None:
        return (a_phase, None, f"acc mas reciente: {a_phase}")
    if a_res is None:
        return (d_phase, None, f"dist mas reciente: {d_phase}")
    if d_res > a_res:
        return (d_phase, None, f"dist mas reciente ({d_res.date()}) que acc ({a_res.date()}): {d_phase}")
    if a_res > d_res:
        return (a_phase, None, f"acc mas reciente ({a_res.date()}) que dist ({d_res.date()}): {a_phase}")
    return ("CONFLICT", f"ambas resueltas en {d_res.date()}", "misma fecha")


def _basis_for(phase: str, side: str, price: float | None, cb: ClassifiedBreak) -> str:
    if phase == "DISTRIBUTION":
        return f"dist vigente: price {price} dentro del rango"
    if phase == "ACCUMULATION":
        return f"acc vigente: price {price} dentro del rango"
    if phase == "MARKDOWN":
        return f"markdown: price {price} < support-tol {cb.range_support}-{cb.range_tol_abs}"
    if phase == "MARKUP":
        return f"markup: price {price} > resistance+tol {cb.range_resistance}+{cb.range_tol_abs}"
    if phase == "RANGE":
        if side == "dist":
            return f"dist fallida: price {price} > resistance+tol {cb.range_resistance}+{cb.range_tol_abs}"
        return f"acc fallida: price {price} < support-tol {cb.range_support}-{cb.range_tol_abs}"
    return phase


def classify_phase_at(df, ticker, as_of, *,
                      dist_kwargs=None, acc_kwargs=None) -> FinalPhase:
    """Ejecuta ambas cascadas hasta as_of y emite la fase final."""
    from src.utils import get_col as _get_col
    dist = classify_distribution_at(df, ticker, as_of, **(dist_kwargs or {}))
    acc = classify_accumulation_at(df, ticker, as_of, **(acc_kwargs or {}))

    # Añadir current_price al evidence (para la regla de vigencia)
    try:
        close = _get_col(df, ticker, "Close")
        current_price = float(close.loc[as_of]) if as_of in close.index else None
    except (KeyError, ValueError):
        current_price = None
    if current_price is not None:
        dist.evidence["current_price"] = current_price
        acc.evidence["current_price"] = current_price

    # Fase de EVENTO (historica, sin vigencia)
    event_phase, event_conflict = _decide_event_phase(dist, acc)

    # Fase ACTUAL (aplica regla de vigencia por precio)
    current_price = dist.evidence.get("current_price")
    phase, conflict, basis = _decide_current_phase(dist, acc, current_price)

    # phase_since: fecha de resolucion del episodio que gano (o None si RANGE/INSUFFICIENT)
    phase_since = None
    if phase == "DISTRIBUTION":
        phase_since = dist.resolution_event_date
    elif phase == "ACCUMULATION":
        phase_since = acc.resolution_event_date

    return FinalPhase(
        ticker=ticker,
        as_of=as_of,
        phase=phase,
        dist_classification=dist.classification,
        acc_classification=acc.classification,
        dist_evidence=dist.evidence,
        acc_evidence=acc.evidence,
        conflict_reason=conflict,
        phase_since=phase_since,
        event_phase=event_phase,
        current_phase_basis=basis,
    )


def classify_phase(df, ticker, *, dist_kwargs=None, acc_kwargs=None) -> FinalPhase:
    """Wrapper con as_of = ultima barra del df."""
    as_of = df.index[-1]
    return classify_phase_at(df, ticker, as_of,
                             dist_kwargs=dist_kwargs, acc_kwargs=acc_kwargs)