"""Tests B1.3 - flujo normativo 10 pasos (v4 seccion 9).

Verifica que P38 no se invoca si PASOS 0-9 no se superan.
Los STOPs del flujo son AdapterError.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.institutional_accumulation.aggregation import catalog_p38_adapter as ca
from src.institutional_accumulation.aggregation import catalog_validator as cv
from src.institutional_accumulation.identity import catalog_key as ck
from src.institutional_accumulation.identity import period_state as ps
from src.institutional_accumulation.identity import target_builder as tb


def _make_universe(rows, version_id="v1", period="2026-03-31"):
    """rows: list of (ticker, figi)."""
    snap = pd.DataFrame(
        [{"radar_ticker": t, "share_class_figi": f, "name": t} for t, f in rows],
        columns=["radar_ticker", "share_class_figi", "name"],
    )
    uids = [ck.compute_snapshot_row_uid(snap.iloc[i], list(snap.columns))
            for i in range(len(rows))]
    keys = ["radar_20260921_" + str(i+1).zfill(4) for i in range(len(rows))]
    mem = pd.DataFrame([
        {"version_id": version_id, "catalog_key": k, "snapshot_row_uid": u,
         "predecessor_row_uid": "", "justification": "test"}
        for k, u in zip(keys, uids)
    ], columns=list(ck.MEMBERSHIP_COLUMNS))
    asg = pd.DataFrame([
        {"catalog_key": k, "assigned_entity_id": "radar_entity_0001",
         "valid_from": "2026-09-21", "valid_to": "", "source": "test",
         "reason": "test"}
        for k in keys
    ], columns=list(ck.ASSIGNMENT_COLUMNS))
    return tb.build_target(snap, mem, asg, version_id=version_id,
                           period_end=period,
                           catalog_version_id="cat", catalog_sha256="a" * 64)


def _same_universe_pair(rows):
    """Crea Q4 y Q1 con las mismas keys (mismo orden -> misma generacion)."""
    u4 = _make_universe(rows)
    u1 = _make_universe(rows)
    return u4, u1

# --- Flujo completo FEASIBLE ---

def test_flujo_completo_feasible():
    rows = [("AAPL", "FIGI_A"), ("MSFT", "FIGI_M")]
    u4, u1 = _same_universe_pair(rows)
    s4 = ps.build_period_state(u4)
    s1 = ps.build_period_state(u1)

    # PASO 3: pairwise
    pairwise = u4.declared_keys & u1.declared_keys
    assert len(pairwise) == 2

    # PASO 5: continuidad
    assert cv.check_continuity(u4, u1) == {}

    # PASO 6: colision
    assert cv.check_economic_collision(u4) == {}
    assert cv.check_economic_collision(u1) == {}

    # PASO 8: full resolution
    assert cv.check_full_resolution(u4, s4) == {}
    assert cv.check_full_resolution(u1, s1) == {}

    # PASO 9: adapter
    t4, t1, r4, r1, feas = ca.catalog_to_p38_targets(
        u4, u1, state_q4=s4, state_q1=s1, pairwise_keys=pairwise,
    )
    assert feas == cv.CoverageFeasibility.FEASIBLE
    assert len(t4) == 2
    assert len(t1) == 2
    assert len(r4) == 2
    assert len(r1) == 2


# --- STOPs del flujo ---

def test_paso3_pairwise_vacio_stop():
    rows = [("AAPL", "FIGI_A")]
    u4, u1 = _same_universe_pair(rows)
    s4 = ps.build_period_state(u4)
    s1 = ps.build_period_state(u1)
    with pytest.raises(ca.AdapterError, match="TARGET_PAIRWISE vacio"):
        ca.catalog_to_p38_targets(u4, u1, state_q4=s4, state_q1=s1,
                                   pairwise_keys=set())


def test_paso6_colision_q4_stop():
    rows = [("AAPL", "FIGI_X"), ("MSFT", "FIGI_X")]
    u4, u1 = _same_universe_pair(rows)
    s4 = ps.build_period_state(u4)
    s1 = ps.build_period_state(u1)
    pairwise = u4.declared_keys & u1.declared_keys
    with pytest.raises(ca.AdapterError, match="CATALOG_ECONOMIC_COLLISION"):
        ca.catalog_to_p38_targets(u4, u1, state_q4=s4, state_q1=s1,
                                   pairwise_keys=pairwise)


def test_paso8_unresolved_stop():
    """Con K UNRESOLVED, el flujo STOP antes de invocar P38.
    El PASO 7 (feasibility) puede dispararse antes que el PASO 8;
    lo que importa es que se lance AdapterError y P38 no se invoque."""
    rows = [("AAPL", "")]
    u4, u1 = _same_universe_pair(rows)
    s4 = ps.build_period_state(u4)
    s1 = ps.build_period_state(u1)
    pairwise = u4.declared_keys & u1.declared_keys
    with pytest.raises(ca.AdapterError):
        ca.catalog_to_p38_targets(u4, u1, state_q4=s4, state_q1=s1,
                                   pairwise_keys=pairwise)


# --- Invariante: P38 no se invoca sin PASO 8 ---

def test_p38_no_invocado_si_unresolved():
    """Si hay K UNRESOLVED en Q4, el adapter no devuelve FEASIBLE."""
    rows = [("AAPL", "FIGI_A"), ("MSFT", "")]
    u4, u1 = _same_universe_pair(rows)
    s4 = ps.build_period_state(u4)
    s1 = ps.build_period_state(u1)
    pairwise = u4.declared_keys & u1.declared_keys
    with pytest.raises(ca.AdapterError):
        ca.catalog_to_p38_targets(u4, u1, state_q4=s4, state_q1=s1,
                                   pairwise_keys=pairwise)


def test_membership_vacio_rechazado():
    """build_target rechaza membership vacio (fail-closed)."""
    snap = pd.DataFrame([], columns=["radar_ticker", "share_class_figi", "name"])
    mem = pd.DataFrame([], columns=list(ck.MEMBERSHIP_COLUMNS))
    asg = pd.DataFrame([], columns=list(ck.ASSIGNMENT_COLUMNS))
    with pytest.raises(tb.BuildTargetError):
        tb.build_target(snap, mem, asg, version_id="v1",
                        period_end="2026-03-31",
                        catalog_version_id="cat", catalog_sha256="a" * 64)