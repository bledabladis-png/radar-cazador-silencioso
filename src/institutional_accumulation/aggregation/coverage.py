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

    Timestamps (B3, A.6.2-bis): los 3 campos temporales son opcionales
    y por defecto None para preservar retrocompatibilidad con
    consumidores existentes (test_p38_contract.py). Su semantica
    contractual esta definida en el contrato semantico seccion 12:
      period_end      fecha de cierre del periodo 13F observado.
      filing_date     fecha de presentacion del filing (SUBMISSION.FILING_DATE).
      knowledge_date  fecha en que la observacion era publicamente conocida
                      (opcion A: == filing_date del accession efectivo;
                       amendments incluidos).
    La coherencia de los 3 (o su ausencia conjunta) es responsabilidad
    del productor; PositionRecord no valida cross-field.
    """
    period: str
    observed_security_key: str
    share_class_figi: Optional[str]
    canonical_security: Optional[str]
    resolution_status: str
    operational_mapping_status: str
    weight: float
    provenance: dict = field(default_factory=dict)
    period_end: Optional[str] = None
    filing_date: Optional[str] = None
    knowledge_date: Optional[str] = None


def aggregate_positions_by_shareclass_figi(records, period, *, return_stats=False):
    """Agrega weights por shareClassFIGI dentro de un periodo.

    F2.4 Opcion 2 AGREG.: solo contribuyen al peso contractual los
    records con operational_mapping_status == VERIFIED.

    records: iterable de PositionRecord.
    period: string identificador del periodo (p.ej. "Q4", "Q1").
    return_stats: si True, devuelve (agg, stats) en lugar de agg.
        Convencion consistente con _filter_canonical (delta_shares.py).

    Devuelve dict {share_class_figi: weight_total} o
    (dict, stats) si return_stats=True. El dict stats tiene:
      n_received               total de records de entrada al filtro
      n_period_mismatch        descartados por period != period arg
      n_missing_figi           descartados por share_class_figi vacio
      n_verified               records VERIFIED que contribuyen
      n_excluded               records con status != VERIFIED
      excluded_by_status       dict {status: count} de no-VERIFIED
      n_temporal_unverified    atajo: excluded_by_status['TEMPORAL_UNVERIFIED']

    Los contadores cuentan RECORDS de entrada, no FIGIs unicos, ni
    CUSIPs, ni pesos. n_received != n_period_mismatch + n_missing_figi
    + n_verified + n_excluded porque los descartes por period/figi
    ocurren antes del conteo por status.
    """
    agg = {}
    n_received = 0
    n_period_mismatch = 0
    n_missing_figi = 0
    n_verified = 0
    excluded_by_status = {}
    for r in records:
        n_received += 1
        if r.period != period:
            n_period_mismatch += 1
            continue
        if not r.share_class_figi:
            n_missing_figi += 1
            continue
        st = r.operational_mapping_status
        if st != "VERIFIED":
            excluded_by_status[st] = excluded_by_status.get(st, 0) + 1
            continue
        n_verified += 1
        agg[r.share_class_figi] = agg.get(r.share_class_figi, 0.0) + float(r.weight)
    if not return_stats:
        return agg
    n_excluded = sum(excluded_by_status.values())
    stats = {
        "n_received": n_received,
        "n_period_mismatch": n_period_mismatch,
        "n_missing_figi": n_missing_figi,
        "n_verified": n_verified,
        "n_excluded": n_excluded,
        "excluded_by_status": dict(excluded_by_status),
        "n_temporal_unverified": excluded_by_status.get("TEMPORAL_UNVERIFIED", 0),
    }
    return agg, stats


def compute_contractual_coverage(target_q4, target_q1, records_q4, records_q1):
    """6 metricas P38 contractuales.

    Contrato seccion 3.2 + 3.3.

    target_q4, target_q1: sets de shareClassFIGI (universo contractual,
                         construido externamente por target_builder).
    records_q4, records_q1: iterables de PositionRecord.

    Devuelve dict con:
      coverage_previous                float | None
      coverage_current                 float | None
      paired_security_coverage         float | None
      paired_weighted_share_coverage   float | None
      coverage_status                  "VALID" | "UNAVAILABLE"
      unmapped_count_previous          int
      unmapped_count_current           int

    A2 c4/5 (auditor Q2 + Q5):
      - coverage_previous = len(res_q4 INTERSECT tq4) / len(tq4).
        Denominador TARGET_Q4 (contractual), no observed.
        None si TARGET_Q4 vacio.
      - coverage_current idem con Q1.
      - paired_security_coverage = None si TARGET_PAIRWISE vacio
        (auditor Q5: fail-closed, no se fabrica 0.0).

    Nota (no tocar en este commit): unmapped_count_previous/current
    siguen midiendo observed sin VERIFIED (no TARGET sin resolver).
    Requiere dictamen para redefinir su semantica contra TARGET.
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

    # A2 c4/5: denominador TARGET (contractual). Interseccion con
    # res_q4 para que FIGIs VERIFIED fuera de TARGET no inflen.
    coverage_previous = (
        len(res_q4 & tq4) / len(tq4) if tq4 else None
    )
    coverage_current = (
        len(res_q1 & tq1) / len(tq1) if tq1 else None
    )

    # TARGET_PAIRWISE = TARGET_Q4 INTERSECT TARGET_Q1.
    target_pairwise = tq4 & tq1

    # PAIRED = TARGET_PAIRWISE INTERSECT res_q4 INTERSECT res_q1.
    paired = target_pairwise & res_q4 & res_q1

    # A2 c4/5 (auditor Q5): TARGET_PAIRWISE vacio -> None (no 0.0).
    paired_security_coverage = (
        len(paired) / len(target_pairwise) if target_pairwise else None
    )

    # Ponderado: w(s) = max(Q4_total(s), Q1_total(s)).
    weights_q4, stats_q4 = aggregate_positions_by_shareclass_figi(
        rq4, "Q4", return_stats=True)
    weights_q1, stats_q1 = aggregate_positions_by_shareclass_figi(
        rq1, "Q1", return_stats=True)

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
        "n_received_q4": stats_q4["n_received"],
        "n_verified_q4": stats_q4["n_verified"],
        "n_temporal_unverified_q4": stats_q4["n_temporal_unverified"],
        "n_excluded_q4": stats_q4["n_excluded"],
        "n_received_q1": stats_q1["n_received"],
        "n_verified_q1": stats_q1["n_verified"],
        "n_temporal_unverified_q1": stats_q1["n_temporal_unverified"],
        "n_excluded_q1": stats_q1["n_excluded"],
        "excluded_by_status_q4": stats_q4["excluded_by_status"],
        "excluded_by_status_q1": stats_q1["excluded_by_status"],
    }
