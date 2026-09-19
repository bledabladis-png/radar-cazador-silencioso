# -*- coding: utf-8 -*-
"""Tests de sec_13f.parser (FA-1.2). Sin red."""
import pandas as pd
import pytest

from src.institutional_accumulation.sec_13f import parser
from src.institutional_accumulation.sec_13f.schema import EXPECTED_COLUMNS


def _write_tsv(path, columns, rows=None):
    """Escribe un TSV sintetico. rows = lista de tuplas."""
    lines = ["\t".join(columns)]
    if rows:
        for row in rows:
            lines.append("\t".join("" if v is None else str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _minimal_row(columns):
    """Devuelve una fila de ejemplo con valores genericos."""
    return tuple("x" for _ in columns)

def test_parse_one_submission_ok(tmp_path):
    cols = EXPECTED_COLUMNS["SUBMISSION"]
    row = ("0001-26-000001", "31-MAR-2026", "13F-HR", "0001", "31-MAR-2026")
    _write_tsv(tmp_path / "SUBMISSION.tsv", cols, [row])
    df = parser.parse_one(tmp_path / "SUBMISSION.tsv", "SUBMISSION")
    assert len(df) == 1
    assert df.iloc[0]["ACCESSION_NUMBER"] == "0001-26-000001"
    assert df.iloc[0]["SUBMISSIONTYPE"] == "13F-HR"


def test_parse_one_fechas_convertidas(tmp_path):
    cols = EXPECTED_COLUMNS["SUBMISSION"]
    row = ("0001", "31-MAR-2026", "13F-HR", "0001", "31-MAR-2026")
    _write_tsv(tmp_path / "SUBMISSION.tsv", cols, [row])
    df = parser.parse_one(tmp_path / "SUBMISSION.tsv", "SUBMISSION")
    assert pd.api.types.is_datetime64_any_dtype(df["FILING_DATE"])
    assert df.iloc[0]["FILING_DATE"].year == 2026
    assert df.iloc[0]["FILING_DATE"].month == 3
    assert df.iloc[0]["FILING_DATE"].day == 31


def test_parse_one_fecha_invalida_coerce_nat(tmp_path):
    cols = EXPECTED_COLUMNS["SUBMISSION"]
    row = ("0001", "NO-VALIDA", "13F-HR", "0001", "31-MAR-2026")
    _write_tsv(tmp_path / "SUBMISSION.tsv", cols, [row])
    df = parser.parse_one(tmp_path / "SUBMISSION.tsv", "SUBMISSION")
    assert pd.isna(df.iloc[0]["FILING_DATE"])


def test_parse_one_dtypes_int64_nullable(tmp_path):
    cols = EXPECTED_COLUMNS["INFOTABLE"]
    # Solo columnas criticas, resto None
    row = (
        "0001", "12345", "ACME", "COM", "123456789", None,
        "1000", "100", "SH", None, "SOLE", None, "0", "0", "100",
    )
    _write_tsv(tmp_path / "INFOTABLE.tsv", cols, [row])
    df = parser.parse_one(tmp_path / "INFOTABLE.tsv", "INFOTABLE")
    assert str(df["INFOTABLE_SK"].dtype) == "Int64"
    assert str(df["VOTING_AUTH_SOLE"].dtype) == "Int64"


def test_parse_one_dtypes_float64_nullable(tmp_path):
    cols = EXPECTED_COLUMNS["INFOTABLE"]
    row = (
        "0001", "12345", "ACME", "COM", "123456789", None,
        "1000.5", "100.25", "SH", None, "SOLE", None, "0", "0", "100",
    )
    _write_tsv(tmp_path / "INFOTABLE.tsv", cols, [row])
    df = parser.parse_one(tmp_path / "INFOTABLE.tsv", "INFOTABLE")
    assert str(df["VALUE"].dtype) == "Float64"
    assert str(df["SSHPRNAMT"].dtype) == "Float64"
    assert float(df.iloc[0]["VALUE"]) == 1000.5

def test_parse_one_tsv_faltante(tmp_path):
    with pytest.raises(FileNotFoundError):
        parser.parse_one(tmp_path / "no_existe.tsv", "SUBMISSION")


def test_parse_one_tsv_no_reconocido(tmp_path):
    p = tmp_path / "X.tsv"
    p.write_text("a\tb\n1\t2\n", encoding="utf-8")
    with pytest.raises(KeyError):
        parser.parse_one(p, "NO_EXISTE")


def test_parse_one_columnas_incorrectas_validate_true(tmp_path):
    _write_tsv(tmp_path / "SUBMISSION.tsv", ["ACCESSION_NUMBER", "OTRA"], [("x", "y")])
    with pytest.raises(ValueError, match="Columnas de SUBMISSION"):
        parser.parse_one(tmp_path / "SUBMISSION.tsv", "SUBMISSION", validate=True)


def test_parse_one_columnas_incorrectas_validate_false(tmp_path):
    _write_tsv(tmp_path / "SUBMISSION.tsv", ["ACCESSION_NUMBER", "OTRA"], [("x", "y")])
    df = parser.parse_one(tmp_path / "SUBMISSION.tsv", "SUBMISSION", validate=False)
    assert "OTRA" in df.columns

def _write_all_7_tsvs(dir_path):
    """Crea los 7 TSVs con 1 fila minima cada uno."""
    for name in ["SUBMISSION", "COVERPAGE", "SUMMARYPAGE",
                 "OTHERMANAGER", "OTHERMANAGER2", "SIGNATURE", "INFOTABLE"]:
        cols = EXPECTED_COLUMNS[name]
        row = tuple("x" for _ in cols)
        _write_tsv(dir_path / (name + ".tsv"), cols, [row])


def test_parse_13f_completo(tmp_path):
    _write_all_7_tsvs(tmp_path)
    result = parser.parse_13f(tmp_path)
    assert len(result) == 7
    for name in ["SUBMISSION", "COVERPAGE", "SUMMARYPAGE",
                 "OTHERMANAGER", "OTHERMANAGER2", "SIGNATURE", "INFOTABLE"]:
        assert name in result
        assert isinstance(result[name], pd.DataFrame)
        assert len(result[name]) == 1


def test_parse_13f_dir_no_existe(tmp_path):
    with pytest.raises(FileNotFoundError):
        parser.parse_13f(tmp_path / "no_existe")


def test_parse_13f_falta_tsv(tmp_path):
    _write_all_7_tsvs(tmp_path)
    (tmp_path / "INFOTABLE.tsv").unlink()
    with pytest.raises(FileNotFoundError, match="TSV faltante"):
        parser.parse_13f(tmp_path)


def test_parse_13f_validate_false(tmp_path):
    _write_all_7_tsvs(tmp_path)
    result = parser.parse_13f(tmp_path, validate=False)
    assert len(result) == 7

def test_parse_one_valor_no_numerico_en_int_coerce_nan(tmp_path):
    """Valor no parseable en columna Int64 -> NaN, no ValueError."""
    cols = EXPECTED_COLUMNS["INFOTABLE"]
    row = (
        "0001", "NO_ES_NUMERO", "ACME", "COM", "123456789", None,
        "1000", "100", "SH", None, "SOLE", None, "0", "0", "100",
    )
    _write_tsv(tmp_path / "INFOTABLE.tsv", cols, [row])
    df = parser.parse_one(tmp_path / "INFOTABLE.tsv", "INFOTABLE")
    assert pd.isna(df.iloc[0]["INFOTABLE_SK"])


def test_parse_one_valor_no_numerico_en_float_coerce_nan(tmp_path):
    """Valor no parseable en columna Float64 -> NaN, no ValueError."""
    cols = EXPECTED_COLUMNS["INFOTABLE"]
    row = (
        "0001", "12345", "ACME", "COM", "123456789", None,
        "NO_NUMERICO", "100", "SH", None, "SOLE", None, "0", "0", "100",
    )
    _write_tsv(tmp_path / "INFOTABLE.tsv", cols, [row])
    df = parser.parse_one(tmp_path / "INFOTABLE.tsv", "INFOTABLE")
    assert pd.isna(df.iloc[0]["VALUE"])