"""B1.3 - Validadores del flujo normativo (v4 secciones 4-7).

check_continuity          (PASO 5) STOP si CONFLICT_FIGI_CHANGE
check_economic_collision  (PASO 6) STOP si CATALOG_ECONOMIC_COLLISION
check_full_resolution     (PASO 8) STOP si identity_status != RESOLVED
                                    o != 1 FIGI por key

Contrato de pureza: no lee/escribe ficheros, no datetime.now(),
deterministas.
"""
from __future__ import annotations


# --- Codigos de error ---
ERR_CONFLICT_FIGI_CHANGE = "CONFLICT_FIGI_CHANGE"
ERR_CATALOG_ECONOMIC_COLLISION = "CATALOG_ECONOMIC_COLLISION"
ERR_FULL_RESOLUTION_FAILED = "FULL_RESOLUTION_FAILED"


# --- Enum de viabilidad de la cadena completa ---
class CoverageFeasibility:
    """Resultado de la cadena de validaciones pre-P38."""
    FEASIBLE = "FEASIBLE"
    UNAVAILABLE = "UNAVAILABLE"

    ALL = (FEASIBLE, UNAVAILABLE)


def check_continuity(universe_q4, universe_q1):
    """PASO 5: verifica continuidad Q4 -> Q1 para K in UNION.

    Si la misma catalog_key declara FIGIs distintos en Q4 y Q1 ->
    CONFLICT_FIGI_CHANGE.

    Parametros:
      universe_q4, universe_q1: TargetUniverse.

    Devuelve dict {catalog_key: error_code} para conflictos.
    """
    errors = {}
    if universe_q4 is None or universe_q1 is None:
        return errors

    keys_union = universe_q4.declared_keys | universe_q1.declared_keys
    for k in keys_union:
        f4 = universe_q4.figi_by_key.get(k)
        f1 = universe_q1.figi_by_key.get(k)
        if f4 is None or f1 is None:
            continue
        if f4 != f1:
            errors[k] = ERR_CONFLICT_FIGI_CHANGE
    return errors


def check_economic_collision(universe):
    """PASO 6: detecta colisiones catalog_key -> FIGI.

    Devuelve {share_class_figi: [catalog_key, ...]} para cada FIGI
    con >1 catalog_key.

    Regla B5: cualquier colision -> CATALOG_ECONOMIC_COLLISION ->
    UNAVAILABLE.
    """
    if universe is None:
        return {}
    by_figi = {}
    for k in universe.declared_keys:
        f = universe.figi_by_key.get(k)
        if not f:
            continue
        by_figi.setdefault(f, []).append(k)
    return {f: ks for f, ks in by_figi.items() if len(ks) > 1}


def check_full_resolution(universe, state):
    """PASO 8: para todo K in universe.declared_keys,
    identity_status == RESOLVED Y exactamente 1 FIGI.

    Devuelve dict {catalog_key: error_code} para fallos.
    """
    errors = {}
    if universe is None or state is None:
        return errors
    for k in universe.declared_keys:
        s = state.get(k)
        if s is None:
            errors[k] = ERR_FULL_RESOLUTION_FAILED
            continue
        if s.identity_status != "RESOLVED":
            errors[k] = ERR_FULL_RESOLUTION_FAILED
            continue
        if not s.figi:
            errors[k] = ERR_FULL_RESOLUTION_FAILED
    return errors