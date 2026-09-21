"""Tests B1.2 - period_state."""
from __future__ import annotations

import pytest

from src.institutional_accumulation.identity import catalog_key as ck
from src.institutional_accumulation.identity import target_builder as tb
from src.institutional_accumulation.identity import period_state as ps
import pandas as pd


def _snapshot(rows):
    return pd.DataFrame(rows, columns=["radar_ticker", "share_class_figi", "name"])


def _membership(version_id, pairs):
    return pd.DataFrame([
        {"version_id": version_id, "catalog_key": k,
         "snapshot_row_uid": uid, "predecessor_row_uid": "",
         "justification": "test"}
        for k, uid in pairs
    ], columns=list(ck.MEMBERSHIP_COLUMNS))


def _assignments(keys):
    return pd.DataFrame([
        {"catalog_key": k, "assigned_entity_id": "radar_entity_0001",
         "valid_from": "2026-09-21", "valid_to": "", "source": "test",
         "reason": "test"}
        for k in keys
    ], columns=list(ck.ASSIGNMENT_COLUMNS))


def _universe(rows):
    """rows: list of (ticker, figi). Keys auto: radar_20260921_NNNN."""
    snap = _snapshot([
        {"radar_ticker": t, "share_class_figi": f, "name": t}
        for t, f in rows
    ])
    uids = [ck.compute_snapshot_row_uid(snap.iloc[i], list(snap.columns))
            for i in range(len(rows))]
    keys = ["radar_20260921_" + str(i+1).zfill(4) for i in range(len(rows))]
    mem = _membership("v1", list(zip(keys, uids)))
    asg = _assignments(keys)
    return tb.build_target(
        snap, mem, asg, version_id="v1", period_end="2026-09-30",
        catalog_version_id="cat", catalog_sha256="a" * 64,
    )

# --- Enums ---

def test_identity_enum():
    assert ps.ALL_IDENTITY_STATUSES == (
        "RESOLVED", "UNRESOLVED", "CONFLICT", "AMBIGUOUS", "NOT_PRESENT",
    )


def test_weight_enum():
    assert ps.ALL_WEIGHT_STATUSES == (
        "RESOLVED_OBSERVED", "ZERO_REPORTED", "NOT_PRESENT",
    )


def test_feasible_weight_set():
    assert ps.FEASIBLE_WEIGHT_STATUSES == frozenset({
        "RESOLVED_OBSERVED", "ZERO_REPORTED",
    })


# --- PeriodState dataclass ---

def test_period_state_valido():
    s = ps.PeriodState(
        identity_status="RESOLVED",
        weight_status="RESOLVED_OBSERVED",
        figi="FIGI_X", ticker="X",
    )
    assert s.identity_status == "RESOLVED"
    assert s.figi == "FIGI_X"


def test_period_state_identity_invalido():
    with pytest.raises(ValueError):
        ps.PeriodState("FOO", "RESOLVED_OBSERVED", "F", "T")


def test_period_state_weight_invalido():
    with pytest.raises(ValueError):
        ps.PeriodState("RESOLVED", "FOO", "F", "T")

# --- build_period_state ---

def test_build_state_por_defecto():
    u = _universe([("AAPL", "FIGI_A"), ("MSFT", "FIGI_M")])
    st = ps.build_period_state(u)
    assert len(st) == 2
    for k in st:
        assert st[k].identity_status == "RESOLVED"
        assert st[k].weight_status == "RESOLVED_OBSERVED"
        assert st[k].figi is not None
        assert st[k].ticker is not None


def test_build_state_figi_vacio_unresolved():
    u = _universe([("AAPL", ""), ("MSFT", "FIGI_M")])
    st = ps.build_period_state(u)
    # La key con figi vacio queda UNRESOLVED
    unresolved = [k for k in st if st[k].identity_status == "UNRESOLVED"]
    assert len(unresolved) == 1
    assert st[unresolved[0]].figi is None


def test_build_state_con_evidence():
    u = _universe([("AAPL", "FIGI_A"), ("MSFT", "FIGI_M")])
    keys = sorted(u.declared_keys)
    figi_ev = {
        keys[0]: {"status": "CONFLICT", "figi": "FIGI_CONFLICT"},
        keys[1]: {"status": "RESOLVED", "figi": "FIGI_M"},
    }
    st = ps.build_period_state(u, figi_evidence=figi_ev)
    assert st[keys[0]].identity_status == "CONFLICT"
    assert st[keys[0]].figi == "FIGI_CONFLICT"
    assert st[keys[1]].identity_status == "RESOLVED"


def test_build_state_weight_evidence():
    u = _universe([("AAPL", "FIGI_A")])
    keys = sorted(u.declared_keys)
    st = ps.build_period_state(u, weight_evidence={keys[0]: "ZERO_REPORTED"})
    assert st[keys[0]].weight_status == "ZERO_REPORTED"

# --- feasible_state ---

def test_feasible_resolved_observed():
    s = ps.PeriodState("RESOLVED", "RESOLVED_OBSERVED", "F", "T")
    assert ps.feasible_state(s) is True


def test_feasible_resolved_zero():
    s = ps.PeriodState("RESOLVED", "ZERO_REPORTED", "F", "T")
    assert ps.feasible_state(s) is True


def test_feasible_unresolved_false():
    s = ps.PeriodState("UNRESOLVED", "RESOLVED_OBSERVED", None, None)
    assert ps.feasible_state(s) is False


def test_feasible_conflict_false():
    s = ps.PeriodState("CONFLICT", "RESOLVED_OBSERVED", "F", "T")
    assert ps.feasible_state(s) is False


def test_feasible_not_present_weight_false():
    s = ps.PeriodState("RESOLVED", "NOT_PRESENT", "F", "T")
    assert ps.feasible_state(s) is False


def test_feasible_none_false():
    assert ps.feasible_state(None) is False


# --- unique_figi ---

def test_unique_figi_resolved():
    s = ps.PeriodState("RESOLVED", "RESOLVED_OBSERVED", "FIGI_X", "T")
    assert ps.unique_figi(s) == "FIGI_X"


def test_unique_figi_conflict_none():
    s = ps.PeriodState("CONFLICT", "RESOLVED_OBSERVED", "FIGI_X", "T")
    assert ps.unique_figi(s) is None


def test_unique_figi_vacio_none():
    s = ps.PeriodState("RESOLVED", "RESOLVED_OBSERVED", "", "T")
    assert ps.unique_figi(s) is None


def test_unique_figi_none_state():
    assert ps.unique_figi(None) is None