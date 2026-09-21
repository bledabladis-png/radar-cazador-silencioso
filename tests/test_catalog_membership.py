"""Tests M - membership + cobertura temporal + predecessor (seccion 3)."""
from __future__ import annotations

import pandas as pd

from src.institutional_accumulation.identity import catalog_key as ck


def _assign(rows):
    return pd.DataFrame([
        {"catalog_key": k, "assigned_entity_id": e,
         "valid_from": vf, "valid_to": vt or "",
         "source": "test", "reason": "test"}
        for k, e, vf, vt in rows
    ], columns=list(ck.ASSIGNMENT_COLUMNS))


def _member(rows):
    return pd.DataFrame([
        {"version_id": vid, "catalog_key": k,
         "snapshot_row_uid": ru, "predecessor_row_uid": pred or "",
         "justification": "test"}
        for vid, k, ru, pred in rows
    ], columns=list(ck.MEMBERSHIP_COLUMNS))


def _manifest(versions):
    return {"snapshots": [
        {"version_id": v, "valid_from": vf, "valid_to": vt}
        for v, vf, vt in versions
    ]}

# --- M-a: snapshot V1 + K1 -> declared_keys={K1} ---

def test_Ma_snapshot_v1_k1():
    a = _assign([("radar_20260921_0001", "radar_entity_0001", "2026-09-21", "")])
    m = _member([("v1", "radar_20260921_0001", "a"*64, None)])
    man = _manifest([("v1", "2026-09-21", None)])
    errors = ck.validate_membership(m, a, man)
    assert errors == {}


# --- M-b: snapshot V2 + K1 (persistencia) -> OK ---

def test_Mb_persistencia():
    a = _assign([("radar_20260921_0001", "radar_entity_0001", "2026-09-21", "")])
    m = _member([("v2", "radar_20260921_0001", "a"*64, None)])
    man = _manifest([("v2", "2026-09-21", None)])
    errors = ck.validate_membership(m, a, man)
    assert errors == {}


# --- M-c: snapshot V2 + K1 + K2 -> OK ---

def test_Mc_dos_keys():
    a = _assign([
        ("radar_20260921_0001", "radar_entity_0001", "2026-09-21", ""),
        ("radar_20260921_0002", "radar_entity_0002", "2026-09-21", ""),
    ])
    m = _member([
        ("v2", "radar_20260921_0001", "a"*64, None),
        ("v2", "radar_20260921_0002", "b"*64, None),
    ])
    man = _manifest([("v2", "2026-09-21", None)])
    errors = ck.validate_membership(m, a, man)
    assert errors == {}


# --- M-d: membership con hueco -> FALLA ---

def test_Md_hueco_key_fuera_assignments():
    a = _assign([("radar_20260921_0001", "radar_entity_0001", "2026-09-21", "")])
    m = _member([("v1", "radar_20260921_9999", "a"*64, None)])
    man = _manifest([("v1", "2026-09-21", None)])
    errors = ck.validate_membership(m, a, man)
    assert errors.get(("v1", "radar_20260921_9999")) == ck.ERR_MISSING_CATALOG_KEY_IN_ASSIGNMENTS

# --- M-e: membership con duplicado de catalog_key -> FALLA ---

def test_Me_duplicado_key_en_snapshot():
    a = _assign([("radar_20260921_0001", "radar_entity_0001", "2026-09-21", "")])
    m = _member([
        ("v1", "radar_20260921_0001", "a"*64, None),
        ("v1", "radar_20260921_0001", "b"*64, None),
    ])
    man = _manifest([("v1", "2026-09-21", None)])
    errors = ck.validate_membership(m, a, man)
    assert errors.get(("v1", "radar_20260921_0001")) == ck.ERR_DUPLICATE_CATALOG_KEY_IN_SNAPSHOT


# --- M-f: membership referencia version_id inexistente -> FALLA ---

def test_Mf_version_inexistente():
    a = _assign([("radar_20260921_0001", "radar_entity_0001", "2026-09-21", "")])
    m = _member([("vNOPE", "radar_20260921_0001", "a"*64, None)])
    man = _manifest([("v1", "2026-09-21", None)])
    errors = ck.validate_membership(m, a, man)
    assert errors.get(("vNOPE", "radar_20260921_0001")) == ck.ERR_MISSING_VERSION_IN_MANIFEST


# --- M-g: reordenacion fisica del CSV no rompe K <-> fila ---

def test_Mg_reordenacion():
    a = _assign([
        ("radar_20260921_0001", "radar_entity_0001", "2026-09-21", ""),
        ("radar_20260921_0002", "radar_entity_0002", "2026-09-21", ""),
    ])
    m = _member([
        ("v1", "radar_20260921_0001", "a"*64, None),
        ("v1", "radar_20260921_0002", "b"*64, None),
    ])
    m_rev = m.iloc[::-1].reset_index(drop=True)
    man = _manifest([("v1", "2026-09-21", None)])
    assert ck.validate_membership(m, a, man) == ck.validate_membership(m_rev, a, man)


# --- M-h: K1 fuera de vigencia del snapshot -> NOT_VALID_FOR_SNAPSHOT ---

def test_Mh_fuera_vigencia():
    a = _assign([
        ("radar_20260101_0001", "radar_entity_0001", "2026-01-01", "2026-06-30"),
    ])
    # snapshot abierto (valid_to=null) con assignment cerrado -> falla
    m = _member([("v1", "radar_20260101_0001", "a"*64, None)])
    man = _manifest([("v1", "2026-09-21", None)])
    errors = ck.validate_membership(m, a, man)
    assert errors.get(("v1", "radar_20260101_0001")) == ck.ERR_CATALOG_KEY_NOT_VALID_FOR_SNAPSHOT

# --- M-i: K1 retired en snapshot posterior -> FALLA ---

def test_Mi_retired_snapshot_posterior():
    a = _assign([
        ("radar_20260101_0001", "radar_entity_0001", "2026-01-01", "2026-06-30"),
    ])
    # snapshot despues del cierre -> assignment no cubre
    m = _member([("v2", "radar_20260101_0001", "a"*64, None)])
    man = _manifest([("v2", "2026-08-01", None)])
    errors = ck.validate_membership(m, a, man)
    assert errors.get(("v2", "radar_20260101_0001")) == ck.ERR_CATALOG_KEY_NOT_VALID_FOR_SNAPSHOT


# --- M-j: filas con contenido identico -> mismo uid ---

def test_Mj_mismo_uid():
    row = {"a": "1", "b": "2"}
    cols = ["a", "b"]
    u1 = ck.compute_snapshot_row_uid(row, cols)
    u2 = ck.compute_snapshot_row_uid(row, cols)
    assert u1 == u2


# --- M-k: predecessor roto -> PREDECESSOR_ROW_UID_BROKEN_CHAIN ---

def test_Mk_predecessor_roto():
    a = _assign([("radar_20260921_0001", "radar_entity_0001", "2026-09-21", "")])
    m = _member([
        ("v1", "radar_20260921_0001", "a"*64, None),
        ("v2", "radar_20260921_0001", "b"*64, "c"*64),  # c no existe
    ])
    man = _manifest([("v1", "2026-09-21", None), ("v2", "2026-09-21", None)])
    errors = ck.validate_membership(m, a, man)
    assert errors.get(("v2", "radar_20260921_0001")) == ck.ERR_PREDECESSOR_ROW_UID_BROKEN_CHAIN


# --- M-m: row_uid cambiado + predecessor valido -> OK ---

def test_Mm_predecessor_valido():
    a = _assign([("radar_20260921_0001", "radar_entity_0001", "2026-09-21", "")])
    m = _member([
        ("v1", "radar_20260921_0001", "a"*64, None),
        ("v2", "radar_20260921_0001", "b"*64, "a"*64),
    ])
    man = _manifest([("v1", "2026-09-21", None), ("v2", "2026-09-21", None)])
    errors = ck.validate_membership(m, a, man)
    assert errors == {}


# --- M-o: snapshot abierto + assignment cerrado -> FALLA ---

def test_Mo_snapshot_abierto_assignment_cerrado():
    a = _assign([
        ("radar_20260101_0001", "radar_entity_0001", "2026-01-01", "2026-06-30"),
    ])
    m = _member([("v1", "radar_20260101_0001", "a"*64, None)])
    man = _manifest([("v1", "2026-09-21", None)])
    errors = ck.validate_membership(m, a, man)
    assert ("v1", "radar_20260101_0001") in errors


# --- M-p: ambos valid_to=NULL -> OK ---

def test_Mp_ambos_abiertos_ok():
    a = _assign([("radar_20260921_0001", "radar_entity_0001", "2026-09-21", "")])
    m = _member([("v1", "radar_20260921_0001", "a"*64, None)])
    man = _manifest([("v1", "2026-09-21", None)])
    errors = ck.validate_membership(m, a, man)
    assert errors == {}