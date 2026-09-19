# -*- coding: utf-8 -*-
"""Tests de sec_13f.identity.relationships (FA-2.3 fix). Sin red."""
import pandas as pd
import pytest

from src.institutional_accumulation.sec_13f.identity import relationships as rel


def _mk_info(rows):
    return pd.DataFrame(rows, columns=[
        "ACCESSION_NUMBER", "INFOTABLE_SK", "OTHERMANAGER", "INVESTMENTDISCRETION",
    ])


def _mk_om2(rows):
    return pd.DataFrame(rows, columns=["ACCESSION_NUMBER", "SEQUENCENUMBER", "CIK"])


def _mk_sub(rows):
    return pd.DataFrame(rows, columns=["ACCESSION_NUMBER", "CIK"])


# ---- constantes ----

def test_estados_definidos():
    for s in rel.ALL_STATUSES:
        assert isinstance(s, str)
    assert rel.STATUS_RESOLVED in rel.ALL_STATUSES
    assert rel.STATUS_NO_REFERENCE in rel.ALL_STATUSES
    assert rel.STATUS_INVALID_ZERO in rel.ALL_STATUSES
    assert rel.STATUS_INVALID_NONNUMERIC in rel.ALL_STATUSES
    assert rel.STATUS_INVALID_OUT_OF_DOMAIN in rel.ALL_STATUSES
    assert rel.STATUS_UNMAPPED_MISSING_OM2 in rel.ALL_STATUSES


def test_seq_max_domain_number_3():
    assert rel.SEQ_MAX_DOMAIN == 999


def test_terminos_prohibidos():
    assert "economic_owner_cik" in rel.FORBIDDEN_TERMS
    assert "owner_cik" in rel.FORBIDDEN_TERMS
    assert "beneficial_owner_cik" in rel.FORBIDDEN_TERMS


# ---- classify_token ----

def test_classify_token_resolved():
    om2_idx = {"A": {1: "CIK1"}}
    status, seq, cik = rel.classify_token("1", "A", om2_idx)
    assert status == rel.STATUS_RESOLVED
    assert seq == 1
    assert cik == "CIK1"


def test_classify_token_unmapped():
    om2_idx = {"A": {2: "CIK2"}}
    status, seq, cik = rel.classify_token("1", "A", om2_idx)
    assert status == rel.STATUS_UNMAPPED_MISSING_OM2
    assert seq == 1
    assert cik is None


def test_classify_token_zero_embebido():
    status, seq, cik = rel.classify_token("0", "A", {})
    assert status == rel.STATUS_INVALID_ZERO
    assert seq is None


def test_classify_token_non_numeric():
    status, seq, cik = rel.classify_token("ARK Advisory", "A", {})
    assert status == rel.STATUS_INVALID_NONNUMERIC
    assert seq is None


def test_classify_token_out_of_domain():
    status, seq, cik = rel.classify_token("1360533", "A", {})
    assert status == rel.STATUS_INVALID_OUT_OF_DOMAIN


def test_classify_token_none_literal():
    status, seq, cik = rel.classify_token("NONE", "A", {})
    assert status == rel.STATUS_NO_REFERENCE


# ---- explode_othermanager_edges ----

def test_explode_no_reference_cero():
    info = _mk_info([("A", 1, "0", "SOLE")])
    om2 = _mk_om2([])
    edges = rel.explode_othermanager_edges(info, om2)
    assert len(edges) == 1
    assert edges.iloc[0]["reference_status"] == rel.STATUS_NO_REFERENCE
    assert pd.isna(edges.iloc[0]["manager_sequence"])


def test_explode_no_reference_none():
    info = _mk_info([("A", 1, "NONE", "SOLE")])
    om2 = _mk_om2([])
    edges = rel.explode_othermanager_edges(info, om2)
    assert len(edges) == 1
    assert edges.iloc[0]["reference_status"] == rel.STATUS_NO_REFERENCE


def test_explode_null():
    info = _mk_info([("A", 1, None, "SOLE")])
    om2 = _mk_om2([])
    edges = rel.explode_othermanager_edges(info, om2)
    assert len(edges) == 1
    assert edges.iloc[0]["reference_status"] == rel.STATUS_NO_REFERENCE


def test_explode_single_resolved():
    info = _mk_info([("A", 1, "5", "SOLE")])
    om2 = _mk_om2([("A", 5, "CIK5")])
    edges = rel.explode_othermanager_edges(info, om2)
    assert len(edges) == 1
    assert edges.iloc[0]["reference_status"] == rel.STATUS_RESOLVED
    assert edges.iloc[0]["manager_sequence"] == 5
    assert edges.iloc[0]["included_manager_cik"] == "CIK5"


def test_explode_multi_manager_tres_edges():
    info = _mk_info([("A", 1, "1,2,3", "DFND")])
    om2 = _mk_om2([("A", 1, "C1"), ("A", 2, "C2"), ("A", 3, "C3")])
    edges = rel.explode_othermanager_edges(info, om2)
    assert len(edges) == 3
    assert set(edges["reference_status"]) == {rel.STATUS_RESOLVED}
    assert set(edges["included_manager_cik"]) == {"C1", "C2", "C3"}


def test_explode_zero_embebido_en_lista():
    info = _mk_info([("A", 1, "0,1,2", "SOLE")])
    om2 = _mk_om2([("A", 1, "C1"), ("A", 2, "C2")])
    edges = rel.explode_othermanager_edges(info, om2)
    assert len(edges) == 3
    statuses = edges["reference_status"].tolist()
    assert rel.STATUS_INVALID_ZERO in statuses
    assert statuses.count(rel.STATUS_RESOLVED) == 2


def test_explode_non_numeric_en_lista():
    info = _mk_info([("A", 1, "1,ARK Advisory,2", "SOLE")])
    om2 = _mk_om2([("A", 1, "C1"), ("A", 2, "C2")])
    edges = rel.explode_othermanager_edges(info, om2)
    assert len(edges) == 3
    statuses = edges["reference_status"].tolist()
    assert rel.STATUS_INVALID_NONNUMERIC in statuses


def test_explode_out_of_domain():
    info = _mk_info([("A", 1, "1360533", "SOLE")])
    om2 = _mk_om2([])
    edges = rel.explode_othermanager_edges(info, om2)
    assert len(edges) == 1
    assert edges.iloc[0]["reference_status"] == rel.STATUS_INVALID_OUT_OF_DOMAIN


def test_explode_unmapped_missing():
    info = _mk_info([("A", 1, "5", "SOLE")])
    om2 = _mk_om2([("A", 1, "C1")])
    edges = rel.explode_othermanager_edges(info, om2)
    assert len(edges) == 1
    assert edges.iloc[0]["reference_status"] == rel.STATUS_UNMAPPED_MISSING_OM2


def test_explode_source_line_id_preservado():
    info = _mk_info([("A", 10, "1,2", "SOLE"), ("A", 11, "3", "SOLE")])
    om2 = _mk_om2([("A", 1, "C1"), ("A", 2, "C2"), ("A", 3, "C3")])
    edges = rel.explode_othermanager_edges(info, om2)
    assert len(edges) == 3
    slids = set(zip(edges["ACCESSION_NUMBER"], edges["INFOTABLE_SK"]))
    assert slids == {("A", "10"), ("A", "11")}


def test_explode_no_divide_economicamente():
    """La expansion multi-edge NO crea filas con VALUE/SSHPRNAMT."""
    info = _mk_info([("A", 1, "1,2,3", "SOLE")])
    om2 = _mk_om2([("A", 1, "C1"), ("A", 2, "C2"), ("A", 3, "C3")])
    edges = rel.explode_othermanager_edges(info, om2)
    # No deben aparecer columnas economicas duplicadas
    assert "VALUE" not in edges.columns
    assert "SSHPRNAMT" not in edges.columns


def test_explode_sin_columnas_lanza():
    with pytest.raises(KeyError):
        rel.explode_othermanager_edges(pd.DataFrame({"X": [1]}), _mk_om2([]))


# ---- compute_edge_metrics ----

def test_metrics_basico():
    info = _mk_info([
        ("A", 1, "1,2", "SOLE"),
        ("A", 2, "0", "SOLE"),
    ])
    om2 = _mk_om2([("A", 1, "C1"), ("A", 2, "C2")])
    edges, metrics = rel.explode_othermanager_edges(info, om2, return_metrics=True)
    assert metrics["source_line_count"] == 2
    assert metrics["unique_source_line_id"] == 2
    assert metrics["invalid_source_line_edges"] == 0
    assert metrics["resolved_edges"] == 2
    assert metrics["no_reference"] == 1


def test_metrics_resolution_rate():
    info = _mk_info([
        ("A", 1, "1,2,3", "SOLE"),
        ("A", 2, "999", "SOLE"),  # 999 no en OM2
    ])
    om2 = _mk_om2([("A", 1, "C1"), ("A", 2, "C2"), ("A", 3, "C3")])
    _, m = rel.explode_othermanager_edges(info, om2, return_metrics=True)
    assert m["resolved_edges"] == 3
    assert m["unmapped_missing_om2"] == 1
    assert 0.0 < m["resolution_rate"] < 1.0


# ---- check_edge_uniqueness ----

def test_check_edge_uniqueness_sin_duplicados():
    info = _mk_info([("A", 1, "1,2", "SOLE")])
    om2 = _mk_om2([("A", 1, "C1"), ("A", 2, "C2")])
    edges = rel.explode_othermanager_edges(info, om2)
    res = rel.check_edge_uniqueness(edges)
    assert res["duplicate_canonical_edges"] == 0
    assert res["unique_canonical_edges"] == 2


# ---- build_canonical_relationship ----

def test_build_canonical_pipeline_completo():
    info = _mk_info([("A", 1, "1,2", "SOLE")])
    om2 = _mk_om2([("A", 1, "C1"), ("A", 2, "C2")])
    sub = _mk_sub([("A", "FM")])
    out = rel.build_canonical_relationship(
        info, om2, sub, report_period="2026-03-31"
    )
    assert len(out) == 2
    assert set(out["included_manager_cik"]) == {"C1", "C2"}
    assert set(out["filing_manager_cik"]) == {"FM"}
    assert set(out["attribution_status"]) == {"PROVISIONAL"}


def test_build_canonical_key_formato():
    info = _mk_info([("A", 1, "1", "SOLE")])
    om2 = _mk_om2([("A", 1, "C1")])
    sub = _mk_sub([("A", "FM")])
    out = rel.build_canonical_relationship(
        info, om2, sub, report_period="2026-03-31"
    )
    assert out.iloc[0]["canonical_reporting_relationship_key"] == "FM|C1|SOLE|2026-03-31"


def test_build_canonical_no_muta():
    info = _mk_info([("A", 1, "1", "SOLE")])
    om2 = _mk_om2([("A", 1, "C1")])
    sub = _mk_sub([("A", "FM")])
    orig = set(info.columns)
    rel.build_canonical_relationship(info, om2, sub, report_period="2026-03-31")
    assert set(info.columns) == orig


def test_build_canonical_sin_terminos_prohibidos():
    info = _mk_info([("A", 1, "1", "SOLE")])
    om2 = _mk_om2([("A", 1, "C1")])
    sub = _mk_sub([("A", "FM")])
    out = rel.build_canonical_relationship(info, om2, sub, report_period="2026-03-31")
    for c in out.columns:
        for t in rel.FORBIDDEN_TERMS:
            assert t not in c.lower()


def test_dfnd_preservado():
    info = _mk_info([
        ("A", 1, "1", "SOLE"),
        ("A", 2, "1", "DFND"),
        ("A", 3, "1", "OTR"),
    ])
    om2 = _mk_om2([("A", 1, "C1")])
    sub = _mk_sub([("A", "FM")])
    out = rel.build_canonical_relationship(info, om2, sub, report_period="2026-03-31")
    assert set(out["discretion_type"]) == {"SOLE", "DFND", "OTR"}


def test_build_filing_manager_index():
    sub = _mk_sub([("A", "FM"), ("B", "FM2")])
    idx = rel.build_filing_manager_index(sub)
    assert len(idx) == 2


def test_build_om2_seq_index():
    om2 = _mk_om2([("A", 1, "C1"), ("A", 2, "C2"), ("B", 1, "C3")])
    idx = rel.build_om2_seq_index(om2)
    assert idx["A"][1] == "C1"
    assert idx["A"][2] == "C2"
    assert idx["B"][1] == "C3"


def test_build_provenance_index():
    om = pd.DataFrame({
        "ACCESSION_NUMBER": ["A"],
        "OTHERMANAGER_SK": [100],
        "CIK": ["C1"],
        "NAME": ["X"],
    })
    idx = rel.build_provenance_index(om)
    assert len(idx) == 1
    assert idx.iloc[0]["reporting_for_cik"] == "C1"


def test_compute_source_line_count():
    info = _mk_info([("A", 1, "1", "SOLE"), ("A", 2, "1", "SOLE")])
    assert rel.compute_source_line_count(info) == 2
