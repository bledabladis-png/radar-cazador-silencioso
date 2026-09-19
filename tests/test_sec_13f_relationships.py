# -*- coding: utf-8 -*-
"""Tests de sec_13f.identity.relationships (FA-2.3). Sin red."""
import pandas as pd
import pytest

from src.institutional_accumulation.sec_13f.identity import relationships as rel


def _mk_sub(rows):
    return pd.DataFrame(rows, columns=["ACCESSION_NUMBER", "CIK"])


def _mk_cov(rows):
    return pd.DataFrame(rows, columns=["ACCESSION_NUMBER", "FILINGMANAGER_NAME"])


def _mk_om(rows):
    return pd.DataFrame(rows, columns=["ACCESSION_NUMBER", "OTHERMANAGER_SK", "CIK", "NAME"])


def _mk_om2(rows):
    return pd.DataFrame(rows, columns=["ACCESSION_NUMBER", "SEQUENCENUMBER", "CIK", "NAME"])


def _mk_info(rows):
    return pd.DataFrame(rows, columns=[
        "ACCESSION_NUMBER", "INFOTABLE_SK", "CUSIP",
        "INVESTMENTDISCRETION", "OTHERMANAGER",
    ])


# ---- contrato ----

def test_constantes_publicas():
    assert "filing_manager_cik" in rel.CANONICAL_KEY_COLUMNS
    assert "included_manager_cik" in rel.CANONICAL_KEY_COLUMNS
    assert "discretion_type" in rel.CANONICAL_KEY_COLUMNS
    assert "report_period" in rel.CANONICAL_KEY_COLUMNS
    assert rel.PROVISIONAL_FLAG is True


def test_terminos_prohibidos_en_constante():
    assert "economic_owner_cik" in rel.FORBIDDEN_TERMS
    assert "owner_cik" in rel.FORBIDDEN_TERMS
    assert "beneficial_owner_cik" in rel.FORBIDDEN_TERMS


# ---- build_filing_manager_index ----

def test_filing_manager_index_basico():
    sub = _mk_sub([("A1", "111"), ("A2", "222")])
    cov = _mk_cov([("A1", "Name1"), ("A2", "Name2")])
    idx = rel.build_filing_manager_index(sub, cov)
    assert len(idx) == 2
    assert set(idx["filing_manager_cik"]) == {"111", "222"}


def test_filing_manager_index_sin_accession_lanza():
    with pytest.raises(KeyError, match="ACCESSION_NUMBER"):
        rel.build_filing_manager_index(pd.DataFrame({"X": [1]}), pd.DataFrame())


def test_filing_manager_index_sin_cik_lanza():
    with pytest.raises(KeyError, match="CIK"):
        rel.build_filing_manager_index(pd.DataFrame({"ACCESSION_NUMBER": ["A"]}), pd.DataFrame())


def test_filing_manager_index_dedup():
    sub = _mk_sub([("A1", "111"), ("A1", "111"), ("A2", "222")])
    cov = _mk_cov([("A1", "N1"), ("A2", "N2")])
    idx = rel.build_filing_manager_index(sub, cov)
    assert len(idx) == 2


# ---- build_included_manager_index ----

def test_included_manager_index_basico():
    om = _mk_om([("A1", 100, "555", "Sub1"), ("A1", 101, "666", "Sub2")])
    idx = rel.build_included_manager_index(om)
    assert len(idx) == 2
    assert set(idx["included_manager_cik"]) == {"555", "666"}


def test_included_manager_index_sin_columnas_lanza():
    with pytest.raises(KeyError, match="OTHERMANAGER_SK"):
        rel.build_included_manager_index(pd.DataFrame({"ACCESSION_NUMBER": ["A"], "CIK": ["1"]}))


def test_included_manager_index_omite_sk_no_numerico():
    om = _mk_om([("A1", "NO_NUM", "555", "x"), ("A1", 100, "666", "y")])
    idx = rel.build_included_manager_index(om)
    assert len(idx) == 1
    assert idx.iloc[0]["included_manager_cik"] == "666"


# ---- build_provenance_index ----

def test_provenance_index_basico():
    om2 = _mk_om2([("A1", 1, "999", "Super1")])
    idx = rel.build_provenance_index(om2)
    assert len(idx) == 1
    assert idx.iloc[0]["supermanager_cik"] == "999"


def test_provenance_index_sin_columnas_lanza():
    with pytest.raises(KeyError, match="SEQUENCENUMBER"):
        rel.build_provenance_index(pd.DataFrame({"ACCESSION_NUMBER": ["A"], "CIK": ["1"]}))


# ---- assign_canonical_key ----

def test_assign_basico_sin_submanager():
    sub = _mk_sub([("A1", "111")])
    cov = _mk_cov([("A1", "x")])
    om = _mk_om([])
    info = _mk_info([("A1", 1, "C1", "SOLE", None)])
    fidx = rel.build_filing_manager_index(sub, cov)
    iidx = rel.build_included_manager_index(om)
    out = rel.assign_canonical_key(info, fidx, iidx, report_period="2026-03-31")
    assert out.iloc[0]["filing_manager_cik"] == "111"
    assert pd.isna(out.iloc[0]["included_manager_cik"])
    assert out.iloc[0]["discretion_type"] == "SOLE"
    assert out.iloc[0]["attribution_status"] == "PROVISIONAL"


def test_assign_con_submanager():
    sub = _mk_sub([("A1", "111")])
    cov = _mk_cov([("A1", "x")])
    om = _mk_om([("A1", 100, "555", "Sub")])
    info = _mk_info([("A1", 1, "C1", "DFND", 100)])
    fidx = rel.build_filing_manager_index(sub, cov)
    iidx = rel.build_included_manager_index(om)
    out = rel.assign_canonical_key(info, fidx, iidx, report_period="2026-03-31")
    assert out.iloc[0]["included_manager_cik"] == "555"
    assert out.iloc[0]["discretion_type"] == "DFND"


def test_canonical_key_formato():
    sub = _mk_sub([("A1", "111")])
    cov = _mk_cov([("A1", "x")])
    om = _mk_om([("A1", 100, "555", "Sub")])
    info = _mk_info([("A1", 1, "C1", "SOLE", 100)])
    fidx = rel.build_filing_manager_index(sub, cov)
    iidx = rel.build_included_manager_index(om)
    out = rel.assign_canonical_key(info, fidx, iidx, report_period="2026-03-31")
    key = out.iloc[0]["canonical_reporting_relationship_key"]
    assert key == "111|555|SOLE|2026-03-31"


def test_canonical_key_sin_included_null():
    sub = _mk_sub([("A1", "111")])
    cov = _mk_cov([("A1", "x")])
    om = _mk_om([])
    info = _mk_info([("A1", 1, "C1", "SOLE", None)])
    fidx = rel.build_filing_manager_index(sub, cov)
    iidx = rel.build_included_manager_index(om)
    out = rel.assign_canonical_key(info, fidx, iidx, report_period="2026-03-31")
    key = out.iloc[0]["canonical_reporting_relationship_key"]
    assert key == "111||SOLE|2026-03-31"


def test_assign_omite_terminos_prohibidos_en_columnas():
    """Ninguna columna de salida puede contener terminos prohibidos."""
    sub = _mk_sub([("A1", "111")])
    cov = _mk_cov([("A1", "x")])
    om = _mk_om([])
    info = _mk_info([("A1", 1, "C1", "SOLE", None)])
    fidx = rel.build_filing_manager_index(sub, cov)
    iidx = rel.build_included_manager_index(om)
    out = rel.assign_canonical_key(info, fidx, iidx, report_period="2026-03-31")
    for col in out.columns:
        for term in rel.FORBIDDEN_TERMS:
            assert term not in col.lower(), "columna con termino prohibido: " + col


def test_assign_no_muta_original():
    sub = _mk_sub([("A1", "111")])
    cov = _mk_cov([("A1", "x")])
    om = _mk_om([])
    info = _mk_info([("A1", 1, "C1", "SOLE", None)])
    fidx = rel.build_filing_manager_index(sub, cov)
    iidx = rel.build_included_manager_index(om)
    orig_cols = set(info.columns)
    rel.assign_canonical_key(info, fidx, iidx, report_period="2026-03-31")
    assert set(info.columns) == orig_cols


def test_assign_sin_investmentdiscretion_lanza():
    sub = _mk_sub([("A1", "111")])
    cov = _mk_cov([("A1", "x")])
    om = _mk_om([])
    info = pd.DataFrame({"ACCESSION_NUMBER": ["A1"], "OTHERMANAGER": [None]})
    fidx = rel.build_filing_manager_index(sub, cov)
    iidx = rel.build_included_manager_index(om)
    with pytest.raises(KeyError, match="INVESTMENTDISCRETION"):
        rel.assign_canonical_key(info, fidx, iidx, report_period="2026-03-31")


def test_attribution_status_es_provisional():
    sub = _mk_sub([("A1", "111")])
    cov = _mk_cov([("A1", "x")])
    om = _mk_om([])
    info = _mk_info([("A1", 1, "C1", "SOLE", None)])
    fidx = rel.build_filing_manager_index(sub, cov)
    iidx = rel.build_included_manager_index(om)
    out = rel.assign_canonical_key(info, fidx, iidx, report_period="2026-03-31")
    assert out["attribution_status"].unique().tolist() == ["PROVISIONAL"]


def test_discretion_dfnd_preservado():
    """Dictamen: DFND no se colapsa con SOLE ni OTR."""
    sub = _mk_sub([("A1", "111")])
    cov = _mk_cov([("A1", "x")])
    om = _mk_om([])
    info = _mk_info([
        ("A1", 1, "C1", "SOLE", None),
        ("A1", 2, "C2", "DFND", None),
        ("A1", 3, "C3", "OTR",  None),
    ])
    fidx = rel.build_filing_manager_index(sub, cov)
    iidx = rel.build_included_manager_index(om)
    out = rel.assign_canonical_key(info, fidx, iidx, report_period="2026-03-31")
    assert set(out["discretion_type"]) == {"SOLE", "DFND", "OTR"}
