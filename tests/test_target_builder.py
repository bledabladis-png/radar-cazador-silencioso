"""Tests B1.2 - target_builder."""
from __future__ import annotations

import pandas as pd
import pytest

from src.institutional_accumulation.identity import catalog_key as ck
from src.institutional_accumulation.identity import target_builder as tb


def _snapshot(rows):
    return pd.DataFrame(
        rows,
        columns=["radar_ticker", "share_class_figi", "name"],
    )


def _membership(version_id, pairs):
    """pairs: list of (catalog_key, snapshot_row_uid)."""
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


def _build(snapshot, membership, assignments, vid="v1", period="2026-09-30"):
    return tb.build_target(
        snapshot, membership, assignments,
        version_id=vid, period_end=period,
        catalog_version_id="cat_v1", catalog_sha256="a" * 64,
    )

# --- Basico ---

def test_build_target_basico():
    snap = _snapshot([{"radar_ticker": "AAPL",
                       "share_class_figi": "BBG001S5N8V8",
                       "name": "APPLE"}])
    uid = ck.compute_snapshot_row_uid(snap.iloc[0], list(snap.columns))
    mem = _membership("v1", [("radar_20260921_0001", uid)])
    asg = _assignments(["radar_20260921_0001"])
    u = _build(snap, mem, asg)
    assert u.declared_keys == {"radar_20260921_0001"}
    assert u.ticker_by_key["radar_20260921_0001"] == "AAPL"
    assert u.figi_by_key["radar_20260921_0001"] == "BBG001S5N8V8"
    assert u.row_uid_by_key["radar_20260921_0001"] == uid
    assert u.key_by_row_uid[uid] == "radar_20260921_0001"


def test_build_target_version_metadata():
    snap = _snapshot([{"radar_ticker": "X", "share_class_figi": "Y", "name": "n"}])
    uid = ck.compute_snapshot_row_uid(snap.iloc[0], list(snap.columns))
    mem = _membership("20260921_01", [("radar_20260921_0001", uid)])
    asg = _assignments(["radar_20260921_0001"])
    u = tb.build_target(snap, mem, asg, version_id="20260921_01",
                        period_end="2026-09-30",
                        catalog_version_id="cat1", catalog_sha256="b" * 64)
    assert u.version_id == "20260921_01"
    assert u.period_end == "2026-09-30"
    assert u.catalog_version_id == "cat1"
    assert u.catalog_sha256 == "b" * 64

# --- Errores ---

def test_membership_no_cubre_uid():
    snap = _snapshot([{"radar_ticker": "X", "share_class_figi": "Y", "name": "n"}])
    mem = _membership("v1", [("radar_20260921_0001", "c" * 64)])
    asg = _assignments(["radar_20260921_0001"])
    with pytest.raises(tb.BuildTargetError, match="no cubre"):
        _build(snap, mem, asg)


def test_catalog_key_no_en_assignments():
    snap = _snapshot([{"radar_ticker": "X", "share_class_figi": "Y", "name": "n"}])
    uid = ck.compute_snapshot_row_uid(snap.iloc[0], list(snap.columns))
    mem = _membership("v1", [("radar_20260921_0001", uid)])
    asg = _assignments(["radar_20260921_9999"])
    with pytest.raises(tb.BuildTargetError, match="no existe en assignments"):
        _build(snap, mem, asg)


def test_membership_vacio_para_version():
    snap = _snapshot([{"radar_ticker": "X", "share_class_figi": "Y", "name": "n"}])
    mem = _membership("v1", [("radar_20260921_0001", "c" * 64)])
    asg = _assignments(["radar_20260921_0001"])
    with pytest.raises(tb.BuildTargetError, match="vac"):
        _build(snap, mem, asg, vid="vNOPE")

def test_schema_error_snapshot_sin_columnas():
    snap = pd.DataFrame([{"name": "x"}], columns=["name"])
    mem = _membership("v1", [("radar_20260921_0001", "c" * 64)])
    asg = _assignments(["radar_20260921_0001"])
    with pytest.raises(ValueError):
        _build(snap, mem, asg)


def test_vinculacion_por_uid_no_por_posicion():
    snap = _snapshot([
        {"radar_ticker": "AAPL", "share_class_figi": "FIGI_A", "name": "A"},
        {"radar_ticker": "MSFT", "share_class_figi": "FIGI_M", "name": "M"},
    ])
    uid_a = ck.compute_snapshot_row_uid(snap.iloc[0], list(snap.columns))
    uid_m = ck.compute_snapshot_row_uid(snap.iloc[1], list(snap.columns))
    mem = _membership("v1", [
        ("radar_20260921_0001", uid_a),
        ("radar_20260921_0002", uid_m),
    ])
    asg = _assignments(["radar_20260921_0001", "radar_20260921_0002"])
    u1 = _build(snap, mem, asg)
    u2 = _build(snap.iloc[::-1].reset_index(drop=True), mem, asg)
    assert u1.ticker_by_key == u2.ticker_by_key
    assert u1.declared_keys == u2.declared_keys


def test_target_universe_es_frozen():
    snap = _snapshot([{"radar_ticker": "X", "share_class_figi": "Y", "name": "n"}])
    uid = ck.compute_snapshot_row_uid(snap.iloc[0], list(snap.columns))
    mem = _membership("v1", [("radar_20260921_0001", uid)])
    asg = _assignments(["radar_20260921_0001"])
    u = _build(snap, mem, asg)
    with pytest.raises(Exception):
        u.period_end = "2026-10-01"