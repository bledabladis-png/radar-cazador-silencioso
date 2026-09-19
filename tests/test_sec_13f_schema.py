# -*- coding: utf-8 -*-
"""Tests de sec_13f.schema (FA-1.1)."""
import pytest

from src.institutional_accumulation.sec_13f import schema


def test_expected_files_son_7():
    assert len(schema.EXPECTED_FILES) == 7


def test_expected_columns_tiene_7_claves():
    assert len(schema.EXPECTED_COLUMNS) == 7
    for k in ["SUBMISSION", "COVERPAGE", "SUMMARYPAGE", "OTHERMANAGER",
              "OTHERMANAGER2", "SIGNATURE", "INFOTABLE"]:
        assert k in schema.EXPECTED_COLUMNS


def test_primary_keys_tiene_7_claves():
    assert len(schema.PRIMARY_KEYS) == 7


def test_infotable_primary_key():
    assert schema.PRIMARY_KEYS["INFOTABLE"] == ("ACCESSION_NUMBER", "INFOTABLE_SK")


def test_othermanager_primary_key():
    assert schema.PRIMARY_KEYS["OTHERMANAGER"] == ("ACCESSION_NUMBER", "OTHERMANAGER_SK")


def test_validate_columns_ok():
    errors = schema.validate_columns("SUBMISSION", schema.EXPECTED_COLUMNS["SUBMISSION"])
    assert errors == []


def test_validate_columns_faltantes():
    errors = schema.validate_columns("SUBMISSION", ["ACCESSION_NUMBER"])
    assert any("faltantes" in e for e in errors)


def test_validate_columns_extras():
    cols = schema.EXPECTED_COLUMNS["SUBMISSION"] + ("EXTRA",)
    errors = schema.validate_columns("SUBMISSION", cols)
    assert any("inesperadas" in e for e in errors)


def test_get_expected_columns_sin_extension():
    cols = schema.get_expected_columns("SUBMISSION")
    assert "ACCESSION_NUMBER" in cols


def test_get_expected_columns_con_extension():
    cols = schema.get_expected_columns("SUBMISSION.tsv")
    assert "ACCESSION_NUMBER" in cols


def test_get_expected_columns_inexistente():
    with pytest.raises(KeyError):
        schema.get_expected_columns("NO_EXISTE")


def test_get_primary_key_inexistente():
    with pytest.raises(KeyError):
        schema.get_primary_key("NO_EXISTE")


def test_schema_version():
    assert schema.SCHEMA_VERSION == "1.0"