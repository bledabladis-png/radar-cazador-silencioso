"""P38: Coverage contractual (F2.4 2026-09-20).

Define la API contractual P38:

    PositionRecord                      dataclass tipado (regla #4 F2.4)
    aggregate_positions_by_shareclass_figi  Opcion 2 AGREG.
    compute_contractual_coverage        funcion contractual P38

El TARGET (target_q4, target_q1) se recibe como parametro externo. Este
modulo NO construye TARGET: eso vive en identity/target_builder.py.

Contrato de pureza (spec 11.3):
  - NO leen ficheros.
  - NO escriben ficheros.
  - NO usan datetime.now().
  - Deterministas y testeables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PositionRecord:
    """Registro tipado de una posicion observada (F2.4 regla #4).

    Sustituye las estructuras paralelas
    (target/resolved/canonical/weights) por un record unico.
    """
    period: str
    observed_security_key: str
    share_class_figi: Optional[str]
    canonical_security: Optional[str]
    resolution_status: str
    operational_mapping_status: str
    weight: float
    provenance: dict = field(default_factory=dict)


def aggregate_positions_by_shareclass_figi(records, period):
    """Agrega weights por shareClassFIGI dentro de un periodo.

    F2.4 Opcion 2 AGREG.: solo contribuyen al peso contractual los
    records con operational_mapping_status == VERIFIED.

    records: iterable de PositionRecord.
    period: string identificador del periodo (p.ej. "Q4", "Q1").

    Devuelve dict {share_class_figi: weight_total}.
    """
    agg = {}
    for r in records:
        if r.period != period:
            continue
        if r.operational_mapping_status != "VERIFIED":
            continue
        if not r.share_class_figi:
            continue
        agg[r.share_class_figi] = agg.get(r.share_class_figi, 0.0) + float(r.weight)
    return agg


def compute_contractual_coverage(target_q4, target_q1, records_q4, records_q1):
    """6 metricas P38 contractuales.

    Contrato seccion 3.2 + 3.3.

    target_q4, target_q1: sets de shareClassFIGI (universo contractual,
                         construido externamente por target_builder).
    records_q4, records_q1: iterables de PositionRecord.

    Devuelve dict con:
      coverage_previous                float
      coverage_current                 float
      paired_security_coverage         float
      paired_weighted_share_coverage   float | None
      coverage_status                  "VALID" | "UNAVAILABLE"
      unmapped_count_previous          int
      unmapped_count_current           int
    """
    tq4 = set(target_q4) if target_q4 else set()
    tq1 = set(target_q1) if target_q1 else set()

    rq4 = list(records_q4) if records_q4 else []
    rq1 = list(records_q1) if records_q1 else []

    # Universo observado por periodo (todos los que tienen FIGI).
    all_q4 = {r.share_class_figi for r in rq4 if r.share_class_figi}
    all_q1 = {r.share_class_figi for r in rq1 if r.share_class_figi}

    # Resueltos por periodo (VERIFIED).
    res_q4 = {
        r.share_class_figi for r in rq4
        if r.share_class_figi
        and r.operational_mapping_status == "VERIFIED"
    }
    res_q1 = {
        r.share_class_figi for r in rq1
        if r.share_class_figi
        and r.operational_mapping_status == "VERIFIED"
    }

    coverage_previous = (len(res_q4) / len(all_q4)) if all_q4 else 0.0
    coverage_current = (len(res_q1) / len(all_q1)) if all_q1 else 0.0

    # TARGET_PAIRWISE = TARGET_Q4 INTERSECT TARGET_Q1.
    target_pairwise = tq4 & tq1

    # PAIRED = TARGET_PAIRWISE INTERSECT res_q4 INTERSECT res_q1.
    paired = target_pairwise & res_q4 & res_q1

    paired_security_coverage = (
        len(paired) / len(target_pairwise) if target_pairwise else 0.0
    )

    # Ponderado: w(s) = max(Q4_total(s), Q1_total(s)).
    weights_q4 = aggregate_positions_by_shareclass_figi(rq4, "Q4")
    weights_q1 = aggregate_positions_by_shareclass_figi(rq1, "Q1")

    def _w(figi):
        return max(weights_q4.get(figi, 0.0), weights_q1.get(figi, 0.0))

    denom = sum(_w(f) for f in target_pairwise)
    numer = sum(_w(f) for f in paired)

    if denom > 0:
        paired_weighted = numer / denom
        coverage_status = "VALID"
    else:
        paired_weighted = None
        coverage_status = "UNAVAILABLE"

    # Regla #2 F2.4: unmapped_count es int, no float.
    unmapped_count_previous = len(all_q4) - len(res_q4)
    unmapped_count_current = len(all_q1) - len(res_q1)

    return {
        "coverage_previous": coverage_previous,
        "coverage_current": coverage_current,
        "paired_security_coverage": paired_security_coverage,
        "paired_weighted_share_coverage": paired_weighted,
        "coverage_status": coverage_status,
        "unmapped_count_previous": unmapped_count_previous,
        "unmapped_count_current": unmapped_count_current,
    }
