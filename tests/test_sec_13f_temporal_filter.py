# -*- coding: utf-8 -*-
"""Tests de sec_13f.identity.temporal_filter (FA-2.1). Sin red."""
import pandas as pd
import pytest

from src.institutional_accumulation.sec_13f.identity import temporal_filter as tf


def _mk_submission(rows):
    """rows: list of (accession, periodofreport, submissiontype, cik)."""
    return pd.DataFrame(rows, columns=[
        "ACCESSION_NUMBER", "PERIODOFREPORT", "SUBMISSIONTYPE", "CIK",
    ])


def _mk_table(accessions, col="VALUE"):
    return pd.DataFrame({
        "ACCESSION_NUMBER": accessions,
        col: list(range(len(accessions))),
    })


def _mk_dfs(sub_rows):
    sub = _mk_submission(sub_rows)
    accs = sub["ACCESSION_NUMBER"].tolist()
    return {
        "SUBMISSION": sub,
        "COVERPAGE": _mk_table(accs),
        "SUMMARYPAGE": _mk_table(accs),
        "OTHERMANAGER": _mk_table(accs),
        "OTHERMANAGER2": _mk_table(accs),
        "SIGNATURE": _mk_table(accs),
        "INFOTABLE": _mk_table(accs),
    }


# ---- tests de firma ----

def test_imports_publicos():
    assert tf.CANONICAL_PERIOD_FIELD == "PERIODOFREPORT"
    assert tf.FULL_PERIOD == "2026-03-31"
    assert callable(tf.filter_by_period)


def test_sin_submission_lanza_keyerror():
    with pytest.raises(KeyError, match="SUBMISSION"):
        tf.filter_by_period({}, "2026-03-31")


def test_submission_sin_periodofreport_lanza():
    dfs = {"SUBMISSION": pd.DataFrame({"ACCESSION_NUMBER": ["a"]})}
    with pytest.raises(KeyError, match="PERIODOFREPORT"):
        tf.filter_by_period(dfs, "2026-03-31")


def test_period_invalido_lanza_valueerror():
    dfs = _mk_dfs([("a", "2026-03-31", "13F-HR", "1")])
    with pytest.raises(ValueError, match="period invalido"):
        tf.filter_by_period(dfs, "no-es-fecha")


# ---- filtro basico ----

def test_filtro_conserva_solo_periodo():
    dfs = _mk_dfs([
        ("a1", "2026-03-31", "13F-HR", "1"),
        ("a2", "2026-03-31", "13F-HR", "2"),
        ("a3", "2025-12-31", "13F-HR", "3"),
    ])
    out = tf.filter_by_period(dfs, "2026-03-31")
    assert set(out["SUBMISSION"]["ACCESSION_NUMBER"]) == {"a1", "a2"}
    assert len(out["SUBMISSION"]) == 2


def test_filtro_propaga_a_los_6_tsvs_derivados():
    dfs = _mk_dfs([
        ("a1", "2026-03-31", "13F-HR", "1"),
        ("a2", "2025-12-31", "13F-HR", "2"),
    ])
    out = tf.filter_by_period(dfs, "2026-03-31")
    for name in ["COVERPAGE", "SUMMARYPAGE", "OTHERMANAGER",
                 "OTHERMANAGER2", "SIGNATURE", "INFOTABLE"]:
        assert len(out[name]) == 1, name
        assert out[name]["ACCESSION_NUMBER"].tolist() == ["a1"]


def test_filtro_sin_filings_devuelve_dict_vacio_pero_con_claves():
    dfs = _mk_dfs([("a1", "2025-01-01", "13F-HR", "1")])
    out = tf.filter_by_period(dfs, "2026-03-31")
    for name in ["SUBMISSION", "COVERPAGE", "INFOTABLE"]:
        assert name in out
        assert len(out[name]) == 0


def test_filtro_idempotente():
    dfs = _mk_dfs([("a1", "2026-03-31", "13F-HR", "1")])
    r1 = tf.filter_by_period(dfs, "2026-03-31")
    r2 = tf.filter_by_period(r1, "2026-03-31")
    for name in r1:
        assert len(r1[name]) == len(r2[name])


def test_filtro_no_muta_dfs_originales():
    dfs = _mk_dfs([
        ("a1", "2026-03-31", "13F-HR", "1"),
        ("a2", "2025-12-31", "13F-HR", "2"),
    ])
    n_orig = len(dfs["SUBMISSION"])
    tf.filter_by_period(dfs, "2026-03-31")
    assert len(dfs["SUBMISSION"]) == n_orig


# ---- stats ----

def test_stats_conteos_basicos():
    dfs = _mk_dfs([
        ("a1", "2026-03-31", "13F-HR", "1"),
        ("a2", "2026-03-31", "13F-HR/A", "2"),
        ("a3", "2026-03-31", "13F-NT", "3"),
        ("a4", "2025-12-31", "13F-HR", "4"),
    ])
    _, stats = tf.filter_by_period(dfs, "2026-03-31", return_stats=True)
    assert stats["filings_total"] == 4
    assert stats["filings_in_period"] == 3
    assert stats["filings_outside_period"] == 1
    assert stats["ciks_in_period"] == 3


def test_stats_submissiontype_breakdown():
    dfs = _mk_dfs([
        ("a1", "2026-03-31", "13F-HR", "1"),
        ("a2", "2026-03-31", "13F-HR", "2"),
        ("a3", "2026-03-31", "13F-HR/A", "3"),
        ("a4", "2026-03-31", "13F-NT", "4"),
        ("a5", "2026-03-31", "13F-NT/A", "5"),
    ])
    _, stats = tf.filter_by_period(dfs, "2026-03-31", return_stats=True)
    assert stats["submissiontype_counts"]["13F-HR"] == 2
    assert stats["submissiontype_counts"]["13F-HR/A"] == 1
    assert stats["submissiontype_counts"]["13F-NT"] == 1
    assert stats["submissiontype_counts"]["13F-NT/A"] == 1


def test_return_stats_false_devuelve_solo_dict():
    dfs = _mk_dfs([("a1", "2026-03-31", "13F-HR", "1")])
    out = tf.filter_by_period(dfs, "2026-03-31")
    assert isinstance(out, dict)
    assert "SUBMISSION" in out


# ---- contrato canonico ----

def test_uses_periodofreport_no_reporcalendar():
    """El filtro usa SUBMISSION.PERIODOFREPORT (dictamen Q-A)."""
    sub = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1", "a2"],
        "PERIODOFREPORT": ["2026-03-31", "2026-03-31"],
        "SUBMISSIONTYPE": ["13F-HR", "13F-HR"],
        "CIK": ["1", "2"],
    })
    cov = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1", "a2"],
        "REPORTCALENDARORQUARTER": ["Y", "Y"],
    })
    dfs = {"SUBMISSION": sub, "COVERPAGE": cov}
    out = tf.filter_by_period(dfs, "2026-03-31")
    assert len(out["SUBMISSION"]) == 2
