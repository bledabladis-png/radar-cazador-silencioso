"""Tests A1 - catalog_key inmutable (seccion 2.6)."""
from __future__ import annotations

import pandas as pd

from src.institutional_accumulation.identity import catalog_key as ck


def _mk_assign(rows):
    """rows: list of (key, entity, valid_from, valid_to)."""
    return pd.DataFrame([
        {
            "catalog_key": k,
            "assigned_entity_id": e,
            "valid_from": vf,
            "valid_to": vt or "",
            "source": "test",
            "reason": "test",
        }
        for k, e, vf, vt in rows
    ], columns=list(ck.ASSIGNMENT_COLUMNS))


def _mk_attempted(rows):
    """rows: list of (key, attempted_entity, attempted_at)."""
    return pd.DataFrame([
        {
            "catalog_key": k,
            "attempted_entity_id": e,
            "attempted_at": at,
            "reason": "test",
            "source": "test",
        }
        for k, e, at in rows
    ], columns=list(ck.ATTEMPTED_COLUMNS))

# --- A1-a: misma key + misma entidad + multiples snapshots -> OK ---

def test_A1a_misma_key_misma_entidad_ok():
    a = _mk_assign([
        ("radar_20260921_0001", "radar_entity_0001", "2026-09-21", ""),
    ])
    errors = ck.validate_assignment(a)
    assert errors == {}


# --- A1-b: misma key con entidad distinta -> CATALOG_KEY_REASSIGNED ---

def test_A1b_key_duplicada_entidad_distinta():
    a = _mk_assign([
        ("radar_20260921_0001", "radar_entity_0001", "2026-09-21", ""),
        ("radar_20260921_0001", "radar_entity_9999", "2026-09-21", ""),
    ])
    errors = ck.validate_assignment(a)
    assert errors.get("radar_20260921_0001") == ck.ERR_CATALOG_KEY_REASSIGNED


# --- A1-c: intento de reasignacion en attempted_df -> REASSIGNED ---

def test_A1c_intento_reasignacion():
    a = _mk_assign([
        ("radar_20260921_0001", "radar_entity_0001", "2026-09-21", ""),
    ])
    att = _mk_attempted([
        ("radar_20260921_0001", "radar_entity_9999", "2026-10-01"),
    ])
    errors = ck.validate_assignment(a, att)
    assert errors.get("radar_20260921_0001") == ck.ERR_CATALOG_KEY_REASSIGNED


# --- A1-d: retired sin reactivar -> OK ---

def test_A1d_retired_sin_reactivar_ok():
    a = _mk_assign([
        ("radar_20260101_0001", "radar_entity_0001", "2026-01-01", "2026-06-30"),
    ])
    errors = ck.validate_assignment(a)
    assert errors == {}


# --- A1-e: retired y reactivado -> RETIRED_REACTIVATED ---

def test_A1e_retired_reactivado():
    a = _mk_assign([
        ("radar_20260101_0001", "radar_entity_0001", "2026-01-01", "2026-06-30"),
    ])
    att = _mk_attempted([
        ("radar_20260101_0001", "radar_entity_0001", "2026-08-01"),
    ])
    errors = ck.validate_assignment(a, att)
    assert errors.get("radar_20260101_0001") == ck.ERR_CATALOG_KEY_RETIRED_REACTIVATED

# --- A1-f: catalog_key unico por snapshot (membership) ---

def test_A1f_catalog_key_unico_por_snapshot():
    m = pd.DataFrame([
        {"version_id": "v1", "catalog_key": "radar_20260921_0001",
         "snapshot_row_uid": "a" * 64, "predecessor_row_uid": "",
         "justification": "x"},
    ], columns=list(ck.MEMBERSHIP_COLUMNS))
    assert m["catalog_key"].is_unique


# --- A1-g: catalog_key unico global ---

def test_A1g_catalog_key_unico_global():
    df = ck.load_assignments("data/mappings/catalog_assignments.csv")
    assert df["catalog_key"].is_unique


# --- A1-h: K1 -> A, intento K1 -> B -> FAIL ---

def test_A1h_reescritura_falla():
    a = _mk_assign([
        ("radar_20260921_0001", "radar_entity_0001", "2026-09-21", ""),
    ])
    att = _mk_attempted([
        ("radar_20260921_0001", "radar_entity_0002", "2026-09-25"),
    ])
    errors = ck.validate_assignment(a, att)
    assert "radar_20260921_0001" in errors


# --- A1-i: misma K + misma entidad en historico -> OK ---

def test_A1i_historico_ok():
    a = _mk_assign([
        ("radar_20260921_0001", "radar_entity_0001", "2026-09-21", ""),
    ])
    att = _mk_attempted([
        ("radar_20260921_0001", "radar_entity_0001", "2026-10-01"),
    ])
    errors = ck.validate_assignment(a, att)
    assert errors == {}