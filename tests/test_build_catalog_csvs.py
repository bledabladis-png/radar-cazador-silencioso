"""Tests B1 - generador de CSV del catalogo (invariantes v5)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
ASSIGN = ROOT / "data" / "mappings" / "catalog_assignments.csv"
MEMBER = ROOT / "data" / "mappings" / "catalog_membership.csv"
SNAPSHOT = ROOT / "data" / "mappings" / "catalog_snapshots" / "snapshot_20260922_01.csv"

import sys
sys.path.insert(0, str(ROOT))
from scripts import build_catalog_csvs as gen


KEY_RE = re.compile(r"^radar_\d{8}_\d{4}$")
ENTITY_RE = re.compile(r"^radar_entity_\d{4}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


# --- Existencia y forma ---

def test_assignments_existe():
    assert ASSIGN.exists()


def test_membership_existe():
    assert MEMBER.exists()


def test_assignments_columnas():
    df = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    assert list(df.columns) == list(gen.ASSIGNMENT_COLUMNS)


def test_membership_columnas():
    df = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)
    assert list(df.columns) == list(gen.MEMBERSHIP_COLUMNS)


def test_catalog_key_formato():
    df = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    for k in df["catalog_key"]:
        assert KEY_RE.match(k), "formato invalido: " + k


def test_assigned_entity_id_formato():
    df = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    for e in df["assigned_entity_id"]:
        assert ENTITY_RE.match(e), "formato invalido: " + e


def test_snapshot_row_uid_sha256_completo():
    df = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)
    for u in df["snapshot_row_uid"]:
        assert SHA_RE.match(u), "sha256 invalido: " + u


# --- Invariantes estructurales ---

def test_catalog_key_unico():
    df = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    assert df["catalog_key"].is_unique


def test_assigned_entity_id_unico():
    df = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    assert df["assigned_entity_id"].is_unique


def test_radar_ticker_unico_en_assignments():
    df = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    assert df["radar_ticker"].is_unique


def test_share_class_figi_unico_en_assignments():
    df = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    assert df["share_class_figi"].is_unique


def test_membership_catalog_keys_en_assignments():
    a = set(pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)["catalog_key"])
    m = set(pd.read_csv(MEMBER, dtype=str, keep_default_na=False)["catalog_key"])
    assert m.issubset(a)


def test_membership_version_id_catalog_key_unico():
    df = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)
    assert not df.duplicated(subset=["version_id", "catalog_key"]).any()


def test_membership_version_id_row_uid_unico():
    df = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)
    assert not df.duplicated(subset=["version_id", "snapshot_row_uid"]).any()


def test_valid_to_null():
    df = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    assert (df["valid_to"] == "").all()


def test_justification_no_vacio():
    df = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)
    assert (df["justification"] != "").all()


def test_predecessor_coherente_con_uid():
    """Invariante Modelo 1 (A1 v4 seccion 3.3, dictamen seccion 10).

    Reglas:
      - Si dos filas comparten catalog_key en el CSV, la posterior debe
        tener predecessor_row_uid == snapshot_row_uid de la anterior.
      - Si la primera fila del CSV para una key tiene predecessor no-vacio
        pero apunta a un UID que NO esta en membership, es valido: apunta
        a una version historica fuera del CSV (migracion per-version).
    """
    df = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)
    uids_in_membership = set(df["snapshot_row_uid"].astype(str))
    # version_id mezcla 2 formatos (20260922_01 y 2026-10-01_01).
    # Normalizar quitando guiones antes de ordenar.
    df = df.assign(_vk=df["version_id"].str.replace("-", "", regex=False))
    for key, g in df.groupby("catalog_key"):
        g = g.sort_values("_vk").reset_index(drop=True)
        prev_uid = None
        for i, row in g.iterrows():
            pred = str(row["predecessor_row_uid"]).strip()
            uid = str(row["snapshot_row_uid"]).strip()
            if i == 0:
                # Primera aparicion en CSV. Si tiene predecessor, debe
                # apuntar a un UID que NO este en membership (historico).
                if pred:
                    assert pred not in uids_in_membership, (
                        "primera aparicion con predecessor dentro del CSV: " + key
                    )
            else:
                if prev_uid == uid:
                    assert pred == "", "UID igual debe tener predecessor vacio: " + key
                else:
                    assert pred == prev_uid, "predecessor debe apuntar al UID previo: " + key
            prev_uid = uid


# --- Idempotencia ---

def test_idempotencia():
    df_before_a = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    df_before_m = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)

    gen.main()

    df_after_a = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    df_after_m = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(df_before_a, df_after_a, check_dtype=False)
    pd.testing.assert_frame_equal(df_before_m, df_after_m, check_dtype=False)


# --- Serializacion canonica ---

def test_canonical_serialization_estable():
    df = pd.read_csv(SNAPSHOT, dtype=str, keep_default_na=False)
    row = df.iloc[0]
    cols = list(df.columns)
    u1 = gen.compute_snapshot_row_uid(row, cols)
    u2 = gen.compute_snapshot_row_uid(row, cols)
    assert u1 == u2


def test_canonical_serialization_columna_orden_independiente():
    df = pd.read_csv(SNAPSHOT, dtype=str, keep_default_na=False)
    row = df.iloc[0]
    cols1 = list(df.columns)
    cols2 = list(reversed(cols1))
    assert gen.compute_snapshot_row_uid(row, cols1) == gen.compute_snapshot_row_uid(row, cols2)


def test_canonical_serialization_cambio_valor_cambia_uid():
    df = pd.read_csv(SNAPSHOT, dtype=str, keep_default_na=False)
    row = df.iloc[0].to_dict()
    cols = list(df.columns)
    u1 = gen.compute_snapshot_row_uid(row, cols)
    row["name"] = row["name"] + "_MUTATED"
    u2 = gen.compute_snapshot_row_uid(row, cols)
    assert u1 != u2


# --- Contrato v5: binding no posicional ---

def _mk_asg(rows):
    return pd.DataFrame(rows, columns=list(gen.ASSIGNMENT_COLUMNS))


def _mk_snap(rows):
    cols = ["radar_ticker", "figi", "share_class_figi"]
    return pd.DataFrame(rows, columns=cols)


def test_binding_no_depende_de_posicion():
    """Reordenar snapshot no cambia el binding ticker<->key."""
    asg = _mk_asg([
        {"catalog_key": "radar_20260101_0001", "assigned_entity_id": "radar_entity_0001",
         "radar_ticker": "AAA", "share_class_figi": "FIGI_A",
         "valid_from": "20260101", "valid_to": "", "source": "t", "reason": "t"},
        {"catalog_key": "radar_20260101_0002", "assigned_entity_id": "radar_entity_0002",
         "radar_ticker": "BBB", "share_class_figi": "FIGI_B",
         "valid_from": "20260101", "valid_to": "", "source": "t", "reason": "t"},
    ])
    snap1 = _mk_snap([["AAA", "BBG_A", "FIGI_A"], ["BBB", "BBG_B", "FIGI_B"]])
    snap2 = _mk_snap([["BBB", "BBG_B", "FIGI_B"], ["AAA", "BBG_A", "FIGI_A"]])

    out1 = gen.merge_assignments(asg, snap1, "20260101", "20260101")
    out2 = gen.merge_assignments(asg, snap2, "20260101", "20260101")

    m1 = dict(zip(out1["radar_ticker"], out1["catalog_key"]))
    m2 = dict(zip(out2["radar_ticker"], out2["catalog_key"]))
    assert m1 == m2


def test_crecimiento_preserva_keys():
    """242 -> 255: los keys antiguos se preservan, los nuevos se anaden."""
    asg = _mk_asg([
        {"catalog_key": "radar_20260101_0001", "assigned_entity_id": "radar_entity_0001",
         "radar_ticker": "AAA", "share_class_figi": "FIGI_A",
         "valid_from": "20260101", "valid_to": "", "source": "t", "reason": "t"},
        {"catalog_key": "radar_20260101_0002", "assigned_entity_id": "radar_entity_0002",
         "radar_ticker": "CCC", "share_class_figi": "FIGI_C",
         "valid_from": "20260101", "valid_to": "", "source": "t", "reason": "t"},
    ])
    # Snapshot crece: se inserta BBB entre AAA y CCC (orden alfabetico)
    snap = _mk_snap([
        ["AAA", "BBG_A", "FIGI_A"],
        ["BBB", "BBG_B", "FIGI_B"],   # nuevo
        ["CCC", "BBG_C", "FIGI_C"],
    ])
    out = gen.merge_assignments(asg, snap, "20260201", "20260201")
    assert len(out) == 3
    m = dict(zip(out["radar_ticker"], out["catalog_key"]))
    assert m["AAA"] == "radar_20260101_0001"
    assert m["CCC"] == "radar_20260101_0002"
    assert m["BBB"].startswith("radar_20260201_")


def test_rename_ticker_misma_key():
    """Mismo share_class_figi, ticker distinto -> misma key."""
    asg = _mk_asg([
        {"catalog_key": "radar_20260101_0001", "assigned_entity_id": "radar_entity_0001",
         "radar_ticker": "FB", "share_class_figi": "FIGI_X",
         "valid_from": "20260101", "valid_to": "", "source": "t", "reason": "t"},
    ])
    snap = _mk_snap([["META", "BBG_X", "FIGI_X"]])
    out = gen.merge_assignments(asg, snap, "20260201", "20260201")
    assert len(out) == 1
    assert out.iloc[0]["catalog_key"] == "radar_20260101_0001"
    assert out.iloc[0]["radar_ticker"] == "META"


def test_cambio_share_class_figi_nueva_key():
    """Mismo ticker, figi distinto -> nueva key."""
    asg = _mk_asg([
        {"catalog_key": "radar_20260101_0001", "assigned_entity_id": "radar_entity_0001",
         "radar_ticker": "AAA", "share_class_figi": "FIGI_OLD",
         "valid_from": "20260101", "valid_to": "", "source": "t", "reason": "t"},
    ])
    snap = _mk_snap([["AAA", "BBG_NEW", "FIGI_NEW"]])
    out = gen.merge_assignments(asg, snap, "20260201", "20260201")
    assert len(out) == 2
    keys = set(out["catalog_key"])
    assert "radar_20260101_0001" in keys
    assert "radar_20260201_0001" in keys


# --- Membership acumulativo ---

def test_membership_acumulativo_entre_versiones():
    asg = _mk_asg([
        {"catalog_key": "radar_20260101_0001", "assigned_entity_id": "radar_entity_0001",
         "radar_ticker": "AAA", "share_class_figi": "FIGI_A",
         "valid_from": "20260101", "valid_to": "", "source": "t", "reason": "t"},
    ])
    snap = _mk_snap([["AAA", "BBG_A", "FIGI_A"]])

    mem0 = pd.DataFrame(columns=list(gen.MEMBERSHIP_COLUMNS))

    # primera version
    import unittest.mock as mock
    with mock.patch.object(gen, "_previous_snapshot_version_id", return_value=None):
        m1 = gen.build_membership_for_version(snap, asg, "20260201_01", mem0)
    assert len(m1) == 1
    assert m1.iloc[0]["justification"] == "initial migration"

    # segunda version: misma fila, mismo UID -> sin predecessor
    mem1 = m1.copy()
    snap2 = _mk_snap([["AAA", "BBG_A", "FIGI_A"]])
    # Para esta version, prev_vid debe devolver la version anterior
    with mock.patch.object(gen, "_previous_snapshot_version_id", return_value="20260201_01"):
        m2 = gen.build_membership_for_version(snap2, asg, "20260301_01", mem1)
    assert len(m2) == 1
    assert m2.iloc[0]["predecessor_row_uid"] == ""
    assert m2.iloc[0]["justification"] == "snapshot content unchanged"


def test_membership_version_ya_presente_no_repite():
    asg = _mk_asg([
        {"catalog_key": "radar_20260101_0001", "assigned_entity_id": "radar_entity_0001",
         "radar_ticker": "AAA", "share_class_figi": "FIGI_A",
         "valid_from": "20260101", "valid_to": "", "source": "t", "reason": "t"},
    ])
    snap = _mk_snap([["AAA", "BBG_A", "FIGI_A"]])
    mem0 = pd.DataFrame([{
        "version_id": "20260201_01", "catalog_key": "radar_20260101_0001",
        "radar_ticker": "AAA", "snapshot_row_uid": "x" * 64,
        "predecessor_row_uid": "", "justification": "x",
    }], columns=list(gen.MEMBERSHIP_COLUMNS))
    out = gen.build_membership_for_version(snap, asg, "20260201_01", mem0)
    assert len(out) == 0