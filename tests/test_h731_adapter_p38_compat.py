"""Tests H-73.1 - compatibilidad adapter B1 -> P38 (dictamen #74).

**Alcance (corregido tras auditoria externa 2026-09-21, hallazgo C-02):**
estos tests verifican la COMPATIBILIDAD SEMANTICA entre el adapter
(catalog_p38_adapter) y el consumidor (compute_contractual_coverage):
los records producidos por el adapter son aceptados por P38 con
`operational_mapping_status == VERIFIED`.

**Lo que estos tests NO verifican:**
- Que el adapter derive `operational_mapping_status` del estado
  operacional real. El adapter lo marca VERIFIED incondicionalmente
  (hallazgo H-10.1, ABIERTO).
- Que `coverage_previous` / `paired_*` sean correctas con datos reales.
  Los tests compat_* usan Q4=Q1 (mock), que fuerza 1.0 trivial.
- Que el peso real (SSHPRNAMT) se propague. `PositionRecord.weight=1.0`
  hardcoded (H-10.1).

Esos puntos requieren fix de codigo productivo (adapter + coverage.py +
period_state.py), bloqueado por prohibicion del prompt v7 hasta dictamen
externo. Ver `iae/EXPEDIENTE_A64_FIX.md`.

El bug original H-73.1 (adapter mapeaba weight_status -> operational)
esta corregido. La proteccion contra el bug H-10.1 (adapter marca
VERIFIED sin verificar) queda PENDIENTE de ciclo propio.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.institutional_accumulation.aggregation import catalog_p38_adapter as ca
from src.institutional_accumulation.aggregation import coverage as cov
from src.institutional_accumulation.identity import catalog_key as ck
from src.institutional_accumulation.identity import period_state as ps
from src.institutional_accumulation.identity import target_builder as tb


def _make_universe(rows):
    snap = pd.DataFrame(
        [{"radar_ticker": t, "share_class_figi": f, "name": t} for t, f in rows],
        columns=["radar_ticker", "share_class_figi", "name"],
    )
    uids = [ck.compute_snapshot_row_uid(snap.iloc[i], list(snap.columns))
            for i in range(len(rows))]
    keys = ["radar_20260921_" + str(i+1).zfill(4) for i in range(len(rows))]
    mem = pd.DataFrame([
        {"version_id": "v1", "catalog_key": k, "snapshot_row_uid": u,
         "predecessor_row_uid": "", "justification": "test"}
        for k, u in zip(keys, uids)
    ], columns=list(ck.MEMBERSHIP_COLUMNS))
    asg = pd.DataFrame([
        {"catalog_key": k, "assigned_entity_id": "radar_entity_0001",
         "valid_from": "2026-09-21", "valid_to": "", "source": "test",
         "reason": "test"}
        for k in keys
    ], columns=list(ck.ASSIGNMENT_COLUMNS))
    return tb.build_target(snap, mem, asg, version_id="v1",
                           period_end="2026-03-31",
                           catalog_version_id="cat", catalog_sha256="a" * 64)


# --- H-73.1-a: records del adapter son VERIFIED ---

def test_compat_a_records_del_adapter_son_verified():
    u = _make_universe([("AAPL", "FIGI_A"), ("MSFT", "FIGI_M")])
    st = ps.build_period_state(u)
    keys = set(u.declared_keys)
    _, _, r4, r1, _ = ca.catalog_to_p38_targets(
        u, u, state_q4=st, state_q1=st, pairwise_keys=keys)
    for r in r4 + r1:
        assert r.operational_mapping_status == "VERIFIED", \
            "adapter debe producir VERIFIED, no " + str(r.operational_mapping_status)
        assert r.resolution_status == "CANONICAL"


# --- H-73.1-b: weight_status RESOLVED_OBSERVED NO se filtra como VERIFIED ---

def test_compat_b_weight_status_no_contamina_operational():
    """El bug original: weight_status=RESOLVED_OBSERVED -> P38 filtra
    como no-VERIFIED. El adapter corregido debe ignorar weight_status
    al construir operational_mapping_status."""
    u = _make_universe([("AAPL", "FIGI_A")])
    st = ps.build_period_state(u)
    # Verificar que weight_status es RESOLVED_OBSERVED (default)
    for k in st:
        assert st[k].weight_status == "RESOLVED_OBSERVED"
    keys = set(u.declared_keys)
    _, _, r4, _, _ = ca.catalog_to_p38_targets(
        u, u, state_q4=st, state_q1=st, pairwise_keys=keys)
    # El record NO debe tener weight_status como operational_mapping_status
    for r in r4:
        assert r.operational_mapping_status != "RESOLVED_OBSERVED"
        assert r.operational_mapping_status == "VERIFIED"


# --- H-73.1-c: compatibilidad end-to-end con compute_contractual_coverage ---

def test_compat_c_p38_acepta_records_del_adapter():
    """COMPATIBILIDAD: P38 acepta los records del adapter.

    **Limitacion explicita:** este test usa Q4=Q1 (mismo state), por lo
    que `coverage=1.0` es trivial por construccion. NO valida cobertura
    contractual. Ver H-10.1 en `iae/EXPEDIENTE_A64_FIX.md`.

    El objetivo es unicamente comprobar que P38 no devuelve UNAVAILABLE
    cuando recibe records del adapter (regresion del bug H-73.1)."""
    u = _make_universe([("AAPL", "FIGI_A"), ("MSFT", "FIGI_M")])
    st = ps.build_period_state(u)
    keys = set(u.declared_keys)
    t4, t1, r4, r1, _ = ca.catalog_to_p38_targets(
        u, u, state_q4=st, state_q1=st, pairwise_keys=keys)

    result = cov.compute_contractual_coverage(t4, t1, r4, r1)
    assert result["coverage_status"] == "VALID", \
        "P38 debe aceptar los records del adapter; coverage_status=" \
        + str(result["coverage_status"])
    assert result["coverage_previous"] == 1.0
    assert result["coverage_current"] == 1.0
    assert result["paired_security_coverage"] == 1.0
    assert result["paired_weighted_share_coverage"] == 1.0


def test_compat_d_regresion_bug_original():
    """Regresion del bug: si adapter mapeara weight_status como
    operational_mapping_status, este test fallaria con UNAVAILABLE."""
    u = _make_universe([("AAPL", "FIGI_A")])
    st = ps.build_period_state(u)
    keys = set(u.declared_keys)
    _, _, r4, _, _ = ca.catalog_to_p38_targets(
        u, u, state_q4=st, state_q1=st, pairwise_keys=keys)

    # Inspeccion directa: si operational_mapping_status fuera
    # RESOLVED_OBSERVED, P38 lo filtraria. Debe ser VERIFIED.
    assert all(r.operational_mapping_status == "VERIFIED" for r in r4)
    # Y si construimos un record con weight_status, P38 debe excluirlo
    bad = cov.PositionRecord(
        period="Q4", observed_security_key="x",
        share_class_figi="FIGI_A", canonical_security=None,
        resolution_status="CANONICAL",
        operational_mapping_status="RESOLVED_OBSERVED",
        weight=1.0,
    )
    result_bad = cov.compute_contractual_coverage(
        {"FIGI_A"}, {"FIGI_A"}, [bad], [bad])
    assert result_bad["coverage_status"] == "UNAVAILABLE", \
        "control negativo: weight_status NO es VERIFIED"


# --- H-73.1-e: universe con solo keys RESOLVED ---

def test_compat_e_state_no_resolved_excluido():
    """Si state[k].identity_status != RESOLVED, el record no se
    construye (defensa por si el caller salta PASO 8)."""
    u = _make_universe([("AAPL", "FIGI_A")])
    keys = list(u.declared_keys)
    st = ps.build_period_state(
        u, figi_evidence={keys[0]: {"status": "CONFLICT", "figi": "FIGI_A"}})
    # catalog_to_p38_targets debe rechazar antes de llegar a _records
    with pytest.raises(ca.AdapterError):
        ca.catalog_to_p38_targets(u, u, state_q4=st, state_q1=st,
                                   pairwise_keys=set(keys))

# --- H-10.1: propiedades correctas del adapter (verificables sin fix) ---

def test_h101_adapter_rechaza_pairwise_vacio():
    """Propiedad correcta: con pairwise vacio, el adapter lanza
    AdapterError. Documenta el comportamiento fail-closed.

    Relevancia H-10.1: en el probe original se paso Q4=Q1 para evitar
    este error, produciendo coverage=1.0 trivial. La solucion no es
    evitar el error, es aceptar fail-closed."""
    u = _make_universe([("AAPL", "FIGI_A")])
    st = ps.build_period_state(u)
    with pytest.raises(ca.AdapterError, match="TARGET_PAIRWISE vacio"):
        ca.catalog_to_p38_targets(u, u, state_q4=st, state_q1=st,
                                   pairwise_keys=set())


def test_h101_adapter_rechaza_state_no_resolved():
    """Propiedad correcta: si una K del pairwise tiene identity_status
    != RESOLVED, el adapter rechaza antes de construir records."""
    u = _make_universe([("AAPL", "FIGI_A")])
    keys = list(u.declared_keys)
    st = ps.build_period_state(
        u, figi_evidence={keys[0]: {"status": "CONFLICT", "figi": "FIGI_A"}})
    with pytest.raises(ca.AdapterError):
        ca.catalog_to_p38_targets(u, u, state_q4=st, state_q1=st,
                                   pairwise_keys=set(keys))


def test_h101_adapter_verified_incondicional_documentado():
    """Documenta (no corrige) el comportamiento H-10.1: el adapter
    actual marca operational_mapping_status="VERIFIED" para todo
    record con identity_status==RESOLVED.

    Este test actua como CONTRATO DE COMPORTAMIENTO ACTUAL. Si el fix
    de H-10.1 (dictamen externo) cambia el comportamiento, este test
    debe actualizarse para reflejar la derivacion real.

    NO verifica que el adapter derive VERIFIED del estado operacional:
    el estado operacional no esta propagado en period_state.
    """
    u = _make_universe([("AAPL", "FIGI_A")])
    st = ps.build_period_state(u)  # default: identity=RESOLVED, weight=RESOLVED_OBSERVED
    keys = set(u.declared_keys)
    _, _, r4, _, _ = ca.catalog_to_p38_targets(
        u, u, state_q4=st, state_q1=st, pairwise_keys=keys)
    # Comportamiento ACTUAL (a cambiar por fix H-10.1):
    for r in r4:
        assert r.operational_mapping_status == "VERIFIED", (
            "comportamiento H-10.1: adapter marca VERIFIED incondicionalmente"
        )
    # Nota: cuando se implemente fix, esta afirmacion cambia. El test
    # se reescribira en el ciclo de fix.

