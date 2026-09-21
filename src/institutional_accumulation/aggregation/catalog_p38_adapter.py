"""B1.3 - Adaptador catalogo -> dominio P38 (v4 seccion 4).

Traduce TARGET administrativo (catalog_key) a TARGET economico P38
(share_class_figi) sin modificar la firma de compute_contractual_coverage.

Regla: full targets. Todos los K de TARGET_Q4 y TARGET_Q1 aparecen,
sin filtro por feasibility. La cobertura pairwise la calcula P38.

Contrato de pureza: no lee/escribe ficheros, no datetime.now(),
deterministas.
"""
from __future__ import annotations

from typing import List, Set

from src.institutional_accumulation.aggregation.catalog_validator import (
    CoverageFeasibility,
    ERR_CATALOG_ECONOMIC_COLLISION,
    ERR_CONFLICT_FIGI_CHANGE,
    ERR_FULL_RESOLUTION_FAILED,
    check_continuity,
    check_economic_collision,
    check_full_resolution,
)
from src.institutional_accumulation.aggregation.coverage import PositionRecord


class AdapterError(ValueError):
    """Fallo en las precondiciones del flujo normativo."""


# --- Firma P38: mantiene interfaz intacta ---
def catalog_to_p38_targets(
    universe_q4,
    universe_q1,
    *,
    state_q4,
    state_q1,
    pairwise_keys,
):
    """Traduce TARGET administrativo a TARGET economico P38.

    Precondiciones (PASOS 0-8 del flujo normativo v4 §9):
      - TARGET_PAIRWISE (pairwise_keys) no vacio.
      - check_continuity OK.
      - check_economic_collision Q4+Q1 OK.
      - feasible(K) para K in TARGET_PAIRWISE.
      - FULL RESOLUTION de TARGET_Q4 UNION TARGET_Q1:
        identity_status == RESOLVED y exactamente 1 share_class_figi.

    Devuelve (target_q4_figi, target_q1_figi, records_q4, records_q1,
    CoverageFeasibility.FEASIBLE).

    Lanza AdapterError si alguna precondicion falla.
    """
    # PASO 3: pairwise no vacio
    if not pairwise_keys:
        raise AdapterError("TARGET_PAIRWISE vacio")

    # PASO 5: continuidad (UNION)
    cont = check_continuity(universe_q4, universe_q1)
    if cont:
        raise AdapterError(
            ERR_CONFLICT_FIGI_CHANGE + ": " + ",".join(sorted(cont))
        )

    # PASO 6: colision economica (Q4 + Q1)
    for label, u in (("Q4", universe_q4), ("Q1", universe_q1)):
        coll = check_economic_collision(u)
        if coll:
            raise AdapterError(
                ERR_CATALOG_ECONOMIC_COLLISION + " " + label + ": "
                + ",".join(sorted(coll))
            )

    # PASO 7: feasibility pairwise
    for k in pairwise_keys:
        s4 = state_q4.get(k)
        s1 = state_q1.get(k)
        if not _is_feasible(s4) or not _is_feasible(s1):
            raise AdapterError("K no feasible en pairwise: " + k)

    # PASO 8: full resolution (UNION)
    for label, u, st in (("Q4", universe_q4, state_q4),
                          ("Q1", universe_q1, state_q1)):
        err = check_full_resolution(u, st)
        if err:
            raise AdapterError(
                ERR_FULL_RESOLUTION_FAILED + " " + label + ": "
                + ",".join(sorted(err))
            )

    # PASO 9: construir targets P38 (full, sin filtro)
    target_q4_figi = _figi_set(universe_q4, state_q4)
    target_q1_figi = _figi_set(universe_q1, state_q1)
    records_q4 = _records(universe_q4, state_q4, period="Q4")
    records_q1 = _records(universe_q1, state_q1, period="Q1")

    return (
        target_q4_figi,
        target_q1_figi,
        records_q4,
        records_q1,
        CoverageFeasibility.FEASIBLE,
    )


def _is_feasible(state):
    if state is None:
        return False
    return (
        state.identity_status == "RESOLVED"
        and state.weight_status in ("RESOLVED_OBSERVED", "ZERO_REPORTED")
    )


def _figi_set(universe, state) -> Set[str]:
    out = set()
    for k in universe.declared_keys:
        s = state.get(k)
        if s is None:
            continue
        if s.identity_status == "RESOLVED" and s.figi:
            out.add(s.figi)
    return out


def _records(universe, state, *, period) -> List[PositionRecord]:
    out = []
    for k in sorted(universe.declared_keys):
        s = state.get(k)
        if s is None:
            continue
        out.append(PositionRecord(
            period=period,
            observed_security_key=k,
            share_class_figi=s.figi,
            canonical_security=None,
            resolution_status=s.identity_status,
            operational_mapping_status=s.weight_status,
            weight=1.0,
        ))
    return out