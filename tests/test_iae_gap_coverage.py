"""Tests de caracterizacion para funciones publicas IAE sin cobertura directa.

Cubre el gap detectado el 2026-09-22 (9 funciones publicas sin mencion
en tests). Prioridad alta: extract_sshprnamt_by_figi (A2 c2, alimenta
el peso contractual P38).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.institutional_accumulation.identity import catalog_key as ck
from src.institutional_accumulation.identity import radar_target_catalog as rtc
from src.institutional_accumulation.identity import target_builder as tb
from src.institutional_accumulation.identity import target_universe as tu


# ============================================================
# extract_sshprnamt_by_figi (target_builder.py) - CRITICA
# ============================================================


def test_extract_sshprnamt_by_figi_df_vacio():
    assert tb.extract_sshprnamt_by_figi(None, {}) == {}
    assert tb.extract_sshprnamt_by_figi(pd.DataFrame(), {}) == {}


def test_extract_sshprnamt_by_figi_df_sin_columnas():
    df = pd.DataFrame([{"X": 1}])
    assert tb.extract_sshprnamt_by_figi(df, {"1": "FIGI"}) == {}


def test_extract_sshprnamt_by_figi_caso_feliz():
    df = pd.DataFrame([
        {"CUSIP": "0001", "SSHPRNAMT": "100"},
        {"CUSIP": "0002", "SSHPRNAMT": "200"},
    ])
    figi = {"0001": "FIGI_A", "0002": "FIGI_B"}
    assert tb.extract_sshprnamt_by_figi(df, figi) == {
        "FIGI_A": 100.0, "FIGI_B": 200.0,
    }


def test_extract_sshprnamt_by_figi_agrega_por_figi():
    df = pd.DataFrame([
        {"CUSIP": "0001", "SSHPRNAMT": "100"},
        {"CUSIP": "0002", "SSHPRNAMT": "200"},
    ])
    figi = {"0001": "FIGI_X", "0002": "FIGI_X"}
    assert tb.extract_sshprnamt_by_figi(df, figi) == {"FIGI_X": 300.0}


def test_extract_sshprnamt_by_figi_cusip_no_mapeado_se_ignora():
    df = pd.DataFrame([{"CUSIP": "0001", "SSHPRNAMT": "100"}])
    assert tb.extract_sshprnamt_by_figi(df, {}) == {}


def test_extract_sshprnamt_by_figi_valores_invalidos_descartados():
    df = pd.DataFrame([
        {"CUSIP": "0001", "SSHPRNAMT": None},
        {"CUSIP": "0002", "SSHPRNAMT": "abc"},
        {"CUSIP": "0003", "SSHPRNAMT": "-5"},
        {"CUSIP": "0004", "SSHPRNAMT": "50"},
    ])
    figi = {c: "FIGI_" + c for c in ["0001", "0002", "0003", "0004"]}
    assert tb.extract_sshprnamt_by_figi(df, figi) == {"FIGI_0004": 50.0}


def test_extract_sshprnamt_by_figi_cusip_vacio_se_ignora():
    df = pd.DataFrame([
        {"CUSIP": "", "SSHPRNAMT": "100"},
        {"CUSIP": None, "SSHPRNAMT": "200"},
    ])
    assert tb.extract_sshprnamt_by_figi(df, {"": "FIGI_X"}) == {}


# ============================================================
# catalog_key - validadores
# ============================================================


@pytest.mark.parametrize("value,esperado", [
    ("radar_20260921_0001", True),
    ("radar_20260921_0000", True),
    ("radar_20260921_9999", True),
    ("radar_2026092_0001", False),
    ("radar_20260921_00001", False),
    ("radar_20260921_1", False),
    ("RADAR_20260921_0001", False),
    ("radar-20260921-0001", False),
    ("radar_20260921_000a", False),
    ("", False),
    (None, False),
    (123, False),
])
def test_is_valid_catalog_key(value, esperado):
    assert ck.is_valid_catalog_key(value) is esperado


@pytest.mark.parametrize("value,esperado", [
    ("radar_entity_0001", True),
    ("radar_entity_9999", True),
    ("radar_entity_00001", False),
    ("radar_entity_1", False),
    ("radar_entity_abcd", False),
    ("RADAR_ENTITY_0001", False),
    ("", False),
    (None, False),
])
def test_is_valid_entity_id(value, esperado):
    assert ck.is_valid_entity_id(value) is esperado


def test_is_valid_sha256():
    ok = "a" * 64
    assert ck.is_valid_sha256(ok) is True
    assert ck.is_valid_sha256("0" * 64) is True
    assert ck.is_valid_sha256("A" * 64) is False  # mayusculas
    assert ck.is_valid_sha256("a" * 63) is False
    assert ck.is_valid_sha256("a" * 65) is False
    assert ck.is_valid_sha256("g" * 64) is False
    assert ck.is_valid_sha256("") is False
    assert ck.is_valid_sha256(None) is False


def test_canonical_serialization_ordenado_y_length_prefixed():
    row = {"b": "x", "a": "y"}
    s = ck.canonical_serialization(row, ["b", "a"])
    # Orden alfabetico: a antes que b.
    # Formato real: <len_name>:<name><len_val>:<val>|
    assert s == "1:a1:y|1:b1:x|"


def test_canonical_serialization_columna_ausente_produce_vacio():
    row = {"a": "x"}
    s = ck.canonical_serialization(row, ["a", "b"])
    # Columna b ausente -> valor canonico = "" -> len 0
    assert s == "1:a1:x|1:b0:|"


def test_canonical_serialization_utf8_bytes_length():
    row = {"a": "caf\u00e9"}  # 4 chars, 5 bytes UTF-8
    s = ck.canonical_serialization(row, ["a"])
    assert s == "1:a5:caf\u00e9|"


# ============================================================
# catalog_key - loaders (tmp_path)
# ============================================================


def test_load_membership_ok(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text(
        ",".join(ck.MEMBERSHIP_COLUMNS) + "\n"
        "v1,radar_20260921_0001,uid1,,test\n",
        encoding="utf-8",
    )
    df = ck.load_membership(p)
    assert len(df) == 1


def test_load_membership_falta_columna_raises(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("solo,columnas\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="catalog_membership.csv"):
        ck.load_membership(p)


def test_load_attempted_ok(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(
        ",".join(ck.ATTEMPTED_COLUMNS) + "\n",
        encoding="utf-8",
    )
    df = ck.load_attempted(p)
    assert list(df.columns) == list(ck.ATTEMPTED_COLUMNS)


def test_load_attempted_falta_columna_raises(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="catalog_reassignments_attempted.csv"):
        ck.load_attempted(p)


# ============================================================
# radar_target_catalog - load_radar_tickers
# ============================================================


def test_load_radar_tickers_filtra_sufijos_no_usa(tmp_path):
    cols = pd.MultiIndex.from_tuples([
        ("Close", "AAPL"),
        ("Close", "SPY"),
        ("Close", "^VIX"),
        ("Close", "EURUSD=X"),
        ("Close", "AIR.PA"),
        ("Close", "SAP.DE"),
    ])
    df = pd.DataFrame([[1.0] * len(cols)], columns=cols)
    p = tmp_path / "sp.parquet"
    df.to_parquet(p)
    tickers = rtc.load_radar_tickers(p)
    assert tickers == ["AAPL", "SPY"]


# ============================================================
# target_universe - load_catalog
# ============================================================


def test_load_catalog_ok(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text(
        "radar_ticker,share_class_figi\nAAPL,BBG000B9XRY4\n",
        encoding="utf-8",
    )
    df = tu.load_catalog(p)
    assert len(df) == 1
    assert "radar_ticker" in df.columns


def test_load_catalog_falta_columnas_raises(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("otra\n1\n", encoding="utf-8")
    with pytest.raises(KeyError):
        tu.load_catalog(p)
