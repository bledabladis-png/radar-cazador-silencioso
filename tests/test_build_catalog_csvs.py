"""Tests B1.0 - generador de CSV del catalogo."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Macro_Sectorial")
ASSIGN = ROOT / "data" / "mappings" / "catalog_assignments.csv"
MEMBER = ROOT / "data" / "mappings" / "catalog_membership.csv"
SNAPSHOT = ROOT / "data" / "mappings" / "catalog_snapshots" / "snapshot_20260921_01.csv"

import sys
sys.path.insert(0, str(ROOT))
from scripts import build_catalog_csvs as gen


KEY_RE = re.compile(r"^radar_\d{8}_\d{4}$")
ENTITY_RE = re.compile(r"^radar_entity_\d{4}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def test_assignments_existe():
    assert ASSIGN.exists()


def test_membership_existe():
    assert MEMBER.exists()


def test_assignments_242_filas():
    df = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    assert len(df) == 242


def test_membership_242_filas():
    df = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)
    assert len(df) == 242


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


def test_catalog_key_unico():
    df = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    assert df["catalog_key"].is_unique


def test_assigned_entity_id_unico():
    df = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    assert df["assigned_entity_id"].is_unique


def test_snapshot_row_uid_sha256_completo():
    df = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)
    for u in df["snapshot_row_uid"]:
        assert SHA_RE.match(u), "sha256 invalido: " + u


def test_snapshot_row_uid_unico():
    df = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)
    assert df["snapshot_row_uid"].is_unique


def test_membership_catalog_keys_en_assignments():
    a = set(pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)["catalog_key"])
    m = set(pd.read_csv(MEMBER, dtype=str, keep_default_na=False)["catalog_key"])
    assert m == a


def test_predecessor_null_en_migracion():
    df = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)
    assert (df["predecessor_row_uid"] == "").all()


def test_justification_initial():
    df = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)
    assert (df["justification"] == "initial migration").all()


def test_valid_to_null():
    df = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    assert (df["valid_to"] == "").all()


def test_idempotencia():
    """Re-ejecutar el generador produce el mismo output."""
    df_before_a = pd.read_csv(ASSIGN, dtype=str, keep_default_na=False)
    df_before_m = pd.read_csv(MEMBER, dtype=str, keep_default_na=False)

    # Regenerar
    csv_path, version_id, alta_date = gen._pick_snapshot()
    df = gen.load_snapshot(csv_path)
    assignments = gen.build_assignments(df, alta_date)
    membership = gen.build_membership(df, version_id, assignments)

    pd.testing.assert_frame_equal(df_before_a, assignments, check_dtype=False)
    pd.testing.assert_frame_equal(df_before_m, membership, check_dtype=False)


def test_canonical_serialization_estable():
    """Misma fila -> mismo uid, dos veces."""
    df = pd.read_csv(SNAPSHOT, dtype=str, keep_default_na=False)
    row = df.iloc[0]
    cols = list(df.columns)
    u1 = gen.compute_snapshot_row_uid(row, cols)
    u2 = gen.compute_snapshot_row_uid(row, cols)
    assert u1 == u2


def test_canonical_serialization_columna_orden_independiente():
    """Reordenar columnas no cambia el uid."""
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