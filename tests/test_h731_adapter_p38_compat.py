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


def _state_verified(universe, *, sshprnamt=1000.0):
    """State con VERIFIED + sshprnamt para todas las keys (A2 c3/5)."""
    return ps.build_period_state(
        universe,
        operational_evidence={k: "VERIFIED" for k in universe.declared_keys},
        sshprnamt_evidence={k: sshprnamt for k in universe.declared_keys},
    )


# --- H-73.1-a: records del adapter PROPAGAN operational_mapping_status ---

def test_compat_a_records_del_adapter_propagan_verified():
    """A2 c3/5: con state VERIFIED, el adapter PROPAGA VERIFIED."""
    u = _make_universe([("AAPL", "FIGI_A"), ("MSFT", "FIGI_M")])
    st = _state_verified(u)
    keys = set(u.declared_keys)
    _, _, r4, r1, _ = ca.catalog_to_p38_targets(
        u, u, state_q4=st, state_q1=st, pairwise_keys=keys)
    for r in r4 + r1:
        assert r.operational_mapping_status == "VERIFIED", (
            "adapter debe propagar VERIFIED del state; obtuvo "
            + str(r.operational_mapping_status)
        )
        assert r.resolution_status == "CANONICAL"


# --- H-73.1-b: state default (UNRESOLVED) -> adapter NO inventa VERIFIED ---

def test_compat_b_propaga_no_inventa_operational():
    """A2 c3/5: con state default (UNRESOLVED), el adapter PROPAGA
    UNRESOLVED y NO inventa VERIFIED. weight_status (RESOLVED_OBSERVED)
    es ortogonal y NO contamina operational_mapping_status."""
    u = _make_universe([("AAPL", "FIGI_A")])
    st = ps.build_period_state(u)  # default: operational=UNRESOLVED
    for k in st:
        assert st[k].weight_status == "RESOLVED_OBSERVED"
        assert st[k].operational_mapping_status == "UNRESOLVED"
    keys = set(u.declared_keys)
    _, _, r4, _, _ = ca.catalog_to_p38_targets(
        u, u, state_q4=st, state_q1=st, pairwise_keys=keys)
    for r in r4:
        assert r.operational_mapping_status == "UNRESOLVED", (
            "adapter no debe inventar VERIFIED; state lo dice UNRESOLVED"
        )
        assert r.operational_mapping_status != "RESOLVED_OBSERVED"


# --- H-73.1-c: compatibilidad end-to-end con compute_contractual_coverage ---

def test_compat_c_p38_acepta_records_del_adapter():
    """COMPATIBILIDAD: con state VERIFIED, P38 acepta los records.

    Nota: coverage=1.0 es trivial por construccion (target=observed=
    VERIFIED, weights>0). NO valida denominador TARGET; eso es
    commit 4. Aqui solo comprobamos que P38 no devuelve UNAVAILABLE
    con records VERIFIED del adapter."""
    u = _make_universe([("AAPL", "FIGI_A"), ("MSFT", "FIGI_M")])
    st = _state_verified(u)
    keys = set(u.declared_keys)
    t4, t1, r4, r1, _ = ca.catalog_to_p38_targets(
        u, u, state_q4=st, state_q1=st, pairwise_keys=keys)

    result = cov.compute_contractual_coverage(t4, t1, r4, r1)
    assert result["coverage_status"] == "VALID", (
        "P38 debe aceptar records VERIFIED del adapter; coverage_status="
        + str(result["coverage_status"])
    )
    assert result["coverage_previous"] == 1.0
    assert result["coverage_current"] == 1.0
    assert result["paired_security_coverage"] == 1.0


def test_compat_d_regresion_bug_original():
    """Regresion del bug: si adapter mapeara weight_status como
    operational_mapping_status, este test fallaria con UNAVAILABLE."""
    u = _make_universe([("AAPL", "FIGI_A")])
    st = _state_verified(u)
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


def test_h101_adapter_propaga_operational_mapping_status():
    """A2 c3/5 (H-10.1 CERRADO): el adapter PROPAGA
    operational_mapping_status 1:1 desde el state. Cubre 3 direcciones:

      (a) state VERIFIED -> record VERIFIED
      (b) state TEMPORAL_UNVERIFIED -> record TEMPORAL_UNVERIFIED
      (c) state sin evidencia (default UNRESOLVED) -> record UNRESOLVED

    El caso (c) es el critico (auditor Q3): el adapter no infiere
    VERIFIED; sin evidencia, fail-closed a UNRESOLVED."""
    u = _make_universe([("AAPL", "FIGI_A")])
    keys = set(u.declared_keys)

    # (a) VERIFIED explicito
    st_a = ps.build_period_state(
        u,
        operational_evidence={k: "VERIFIED" for k in keys},
    )
    _, _, r4a, _, _ = ca.catalog_to_p38_targets(
        u, u, state_q4=st_a, state_q1=st_a, pairwise_keys=keys)
    assert all(r.operational_mapping_status == "VERIFIED" for r in r4a)

    # (b) TEMPORAL_UNVERIFIED explicito
    st_b = ps.build_period_state(
        u,
        operational_evidence={k: "TEMPORAL_UNVERIFIED" for k in keys},
    )
    _, _, r4b, _, _ = ca.catalog_to_p38_targets(
        u, u, state_q4=st_b, state_q1=st_b, pairwise_keys=keys)
    assert all(r.operational_mapping_status == "TEMPORAL_UNVERIFIED"
               for r in r4b)

    # (c) sin evidencia -> default UNRESOLVED (fail-closed)
    st_c = ps.build_period_state(u)  # sin operational_evidence
    _, _, r4c, _, _ = ca.catalog_to_p38_targets(
        u, u, state_q4=st_c, state_q1=st_c, pairwise_keys=keys)
    assert all(r.operational_mapping_status == "UNRESOLVED" for r in r4c)


def test_h101_adapter_propaga_weight_desde_sshprnamt():
    """A2 c3/5 (H-07 CERRADO): el adapter PROPAGA el peso contractual
    desde state.sshprnamt. Sin sshprnamt -> weight=0.0 (fail-closed,
    no se inventa peso)."""
    u = _make_universe([("AAPL", "FIGI_A"), ("MSFT", "FIGI_M")])
    keys = set(u.declared_keys)
    k_aapl = None
    k_msft = None
    for k in keys:
        if u.ticker_by_key[k] == "AAPL":
            k_aapl = k
        elif u.ticker_by_key[k] == "MSFT":
            k_msft = k
    assert k_aapl is not None and k_msft is not None

    st = ps.build_period_state(
        u,
        operational_evidence={k: "VERIFIED" for k in keys},
        sshprnamt_evidence={k_aapl: 1500.0, k_msft: None},
    )
    _, _, r4, _, _ = ca.catalog_to_p38_targets(
        u, u, state_q4=st, state_q1=st, pairwise_keys=keys)

    by_key = {r.observed_security_key: r for r in r4}
    assert by_key[k_aapl].weight == 1500.0
    assert by_key[k_msft].weight == 0.0  # fail-closed, no inventar peso

