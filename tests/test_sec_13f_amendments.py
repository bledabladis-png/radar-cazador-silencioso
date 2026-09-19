# -*- coding: utf-8 -*-
"""Tests de sec_13f.identity.amendments (FA-2.4). Sin red."""
import pandas as pd

from src.institutional_accumulation.sec_13f.identity import amendments as am


def _mk_sub(rows):
    return pd.DataFrame(rows, columns=["ACCESSION_NUMBER", "CIK", "PERIODOFREPORT", "FILING_DATE", "SUBMISSIONTYPE"])


def _mk_cov(rows):
    return pd.DataFrame(rows, columns=["ACCESSION_NUMBER", "AMENDMENTNO", "AMENDMENTTYPE", "ISAMENDMENT"])


# ---- constantes ----

def test_strategies_definidas():
    for s in am.ALL_STRATEGIES:
        assert isinstance(s, str)
    assert am.STRATEGY_SINGLE_HR in am.ALL_STRATEGIES
    assert am.STRATEGY_HR_COMPOSITE in am.ALL_STRATEGIES
    assert am.STRATEGY_NOTICE_AMENDED in am.ALL_STRATEGIES


def test_status_definidos():
    for s in am.ALL_STATUSES:
        assert isinstance(s, str)


def test_anomalias_definidas():
    for a in am.ALL_ANOMALIES:
        assert isinstance(a, str)


# ---- order_filings ----

def test_order_original_antes_que_amendment():
    sub = _mk_sub([
        ("A2", "1", "2026-03-31", "2026-05-19", "13F-HR/A"),
        ("A1", "1", "2026-03-31", "2026-05-15", "13F-HR"),
    ])
    cov = _mk_cov([
        ("A2", 1, "RESTATEMENT", "Y"),
        ("A1", None, None, None),
    ])
    out = am.order_filings(sub, cov)
    assert out.iloc[0]["ACCESSION_NUMBER"] == "A1"
    assert out.iloc[1]["ACCESSION_NUMBER"] == "A2"


def test_order_por_amendment_no():
    sub = _mk_sub([
        ("A3", "1", "2026-03-31", "2026-05-20", "13F-HR/A"),
        ("A2", "1", "2026-03-31", "2026-05-15", "13F-HR/A"),
        ("A1", "1", "2026-03-31", "2026-05-11", "13F-HR"),
    ])
    cov = _mk_cov([
        ("A3", 2, "RESTATEMENT", "Y"),
        ("A2", 1, "RESTATEMENT", "Y"),
        ("A1", None, None, None),
    ])
    out = am.order_filings(sub, cov)
    assert out["ACCESSION_NUMBER"].tolist() == ["A1", "A2", "A3"]


def test_order_amendment_no_gana_a_fecha():
    """AMENDMENTNO=2 con fecha anterior debe ir DESPUES de AMENDMENTNO=1."""
    sub = _mk_sub([
        ("A2", "1", "2026-03-31", "2026-05-01", "13F-HR/A"),  # mas antiguo pero AN=2
        ("A3", "1", "2026-03-31", "2026-05-20", "13F-HR/A"),  # mas nuevo pero AN=1
        ("A1", "1", "2026-03-31", "2026-04-30", "13F-HR"),
    ])
    cov = _mk_cov([
        ("A2", 2, "RESTATEMENT", "Y"),
        ("A3", 1, "RESTATEMENT", "Y"),
        ("A1", None, None, None),
    ])
    out = am.order_filings(sub, cov)
    # Orden: A1 (HR), A3 (AN=1), A2 (AN=2)
    assert out["ACCESSION_NUMBER"].tolist() == ["A1", "A3", "A2"]


# ---- detect_base_filing ----

def test_base_hr_unico():
    sub = _mk_sub([("A1", "1", "2026-03-31", "2026-05-15", "13F-HR")])
    cov = _mk_cov([("A1", None, None, None)])
    ordered = am.order_filings(sub, cov)
    grp = ordered[ordered["CIK"] == "1"]
    acc, status = am.detect_base_filing(grp)
    assert acc == "A1"
    assert status == am.STATUS_CANONICAL


def test_base_notice_sin_holdings():
    sub = _mk_sub([("A1", "1", "2026-03-31", "2026-05-15", "13F-NT")])
    cov = _mk_cov([("A1", None, None, None)])
    ordered = am.order_filings(sub, cov)
    acc, status = am.detect_base_filing(ordered)
    assert acc is None
    assert status == am.STATUS_NO_HOLDINGS


def test_base_ambiguo():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-15", "13F-HR"),
        ("A2", "1", "2026-03-31", "2026-05-20", "13F-HR"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("A2", None, None, None),
    ])
    ordered = am.order_filings(sub, cov)
    acc, status = am.detect_base_filing(ordered)
    assert acc is None
    assert status == "AMBIGUOUS_BASE"


# ---- classify_strategy ----

def test_strategy_single_hr():
    sub = _mk_sub([("A1", "1", "2026-03-31", "2026-05-15", "13F-HR")])
    cov = _mk_cov([("A1", None, None, None)])
    ordered = am.order_filings(sub, cov)
    assert am.classify_strategy(ordered) == am.STRATEGY_SINGLE_HR


def test_strategy_single_notice():
    sub = _mk_sub([("A1", "1", "2026-03-31", "2026-05-15", "13F-NT")])
    cov = _mk_cov([("A1", None, None, None)])
    ordered = am.order_filings(sub, cov)
    assert am.classify_strategy(ordered) == am.STRATEGY_SINGLE_NOTICE


def test_strategy_hr_plus_restatement():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-15", "13F-HR"),
        ("A2", "1", "2026-03-31", "2026-05-19", "13F-HR/A"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("A2", 1, "RESTATEMENT", "Y"),
    ])
    ordered = am.order_filings(sub, cov)
    assert am.classify_strategy(ordered) == am.STRATEGY_HR_PLUS_RESTATEMENT


def test_strategy_hr_plus_new_holdings():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-15", "13F-HR"),
        ("A2", "1", "2026-03-31", "2026-05-19", "13F-HR/A"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("A2", 1, "NEW HOLDINGS", "Y"),
    ])
    ordered = am.order_filings(sub, cov)
    assert am.classify_strategy(ordered) == am.STRATEGY_HR_PLUS_NEW_HOLDINGS


def test_strategy_hr_chain_restatement():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-11", "13F-HR"),
        ("A2", "1", "2026-03-31", "2026-05-15", "13F-HR/A"),
        ("A3", "1", "2026-03-31", "2026-05-20", "13F-HR/A"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("A2", 1, "RESTATEMENT", "Y"),
        ("A3", 2, "RESTATEMENT", "Y"),
    ])
    ordered = am.order_filings(sub, cov)
    assert am.classify_strategy(ordered) == am.STRATEGY_HR_CHAIN_RESTATEMENT


def test_strategy_hr_composite():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-08", "13F-HR"),
        ("A2", "1", "2026-03-31", "2026-05-15", "13F-HR/A"),
        ("A3", "1", "2026-03-31", "2026-05-20", "13F-HR/A"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("A2", 1, "RESTATEMENT", "Y"),
        ("A3", 2, "NEW HOLDINGS", "Y"),
    ])
    ordered = am.order_filings(sub, cov)
    assert am.classify_strategy(ordered) == am.STRATEGY_HR_COMPOSITE


def test_strategy_notice_amended():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-15", "13F-NT"),
        ("A2", "1", "2026-03-31", "2026-05-19", "13F-NT/A"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("A2", 1, "NEW HOLDINGS", "Y"),
    ])
    ordered = am.order_filings(sub, cov)
    assert am.classify_strategy(ordered) == am.STRATEGY_NOTICE_AMENDED


# ---- apply_amendments ----

def _mk_dfs(sub, cov, info_rows=None):
    dfs = {
        "SUBMISSION": sub,
        "COVERPAGE": cov,
    }
    if info_rows is not None:
        dfs["INFOTABLE"] = pd.DataFrame(info_rows, columns=["ACCESSION_NUMBER", "INFOTABLE_SK", "CUSIP"])
    return dfs


def test_apply_single_hr():
    sub = _mk_sub([("A1", "1", "2026-03-31", "2026-05-15", "13F-HR")])
    cov = _mk_cov([("A1", None, None, None)])
    info = [("A1", 1, "C1")]
    dfs = _mk_dfs(sub, cov, info)
    res = am.apply_amendments(dfs, period="2026-03-31")
    assert res["canonical_snapshot"]["SUBMISSION"].shape[0] == 1
    assert "A1" in res["applied_accessions"]


def test_apply_restatement_reemplaza():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-15", "13F-HR"),
        ("A2", "1", "2026-03-31", "2026-05-19", "13F-HR/A"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("A2", 1, "RESTATEMENT", "Y"),
    ])
    info = [("A1", 1, "C1"), ("A2", 1, "C1")]
    dfs = _mk_dfs(sub, cov, info)
    res = am.apply_amendments(dfs, period="2026-03-31")
    assert res["applied_accessions"] == {"A2"}


def test_apply_new_holdings_compone():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-15", "13F-HR"),
        ("A2", "1", "2026-03-31", "2026-05-19", "13F-HR/A"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("A2", 1, "NEW HOLDINGS", "Y"),
    ])
    info = [("A1", 1, "C1"), ("A2", 2, "C2")]
    dfs = _mk_dfs(sub, cov, info)
    res = am.apply_amendments(dfs, period="2026-03-31")
    assert res["applied_accessions"] == {"A1", "A2"}


def test_apply_nt_no_contribuye():
    sub = _mk_sub([("A1", "1", "2026-03-31", "2026-05-15", "13F-NT")])
    cov = _mk_cov([("A1", None, None, None)])
    info = [("A1", 1, "C1")]
    dfs = _mk_dfs(sub, cov, info)
    res = am.apply_amendments(dfs, period="2026-03-31")
    assert res["applied_accessions"] == set()
    assert res["canonical_snapshot"]["SUBMISSION"].shape[0] == 0


def test_apply_nt_a_new_holdings_anomalia():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-15", "13F-NT"),
        ("A2", "1", "2026-03-31", "2026-05-19", "13F-NT/A"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("A2", 1, "NEW HOLDINGS", "Y"),
    ])
    dfs = _mk_dfs(sub, cov, [])
    res = am.apply_amendments(dfs, period="2026-03-31")
    assert "A2" not in res["applied_accessions"]
    anoms = res["anomalies"]["anomaly_code"].tolist()
    assert am.ANOMALY_NT_AUGMENTED in anoms


def test_apply_hr_with_amendment_flags_anomalia():
    sub = _mk_sub([("A1", "1", "2026-03-31", "2026-04-14", "13F-HR")])
    cov = _mk_cov([("A1", 1, "NEW HOLDINGS", "Y")])
    info = [("A1", 1, "C1")]
    dfs = _mk_dfs(sub, cov, info)
    res = am.apply_amendments(dfs, period="2026-03-31")
    assert "A1" in res["applied_accessions"]
    anoms = res["anomalies"]["anomaly_code"].tolist()
    assert am.ANOMALY_HR_WITH_AMENDMENT_FLAGS in anoms


def test_apply_ambiguous_amendment_order():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-11", "13F-HR"),
        ("A2", "1", "2026-03-31", "2026-05-15", "13F-HR/A"),
        ("A3", "1", "2026-03-31", "2026-05-16", "13F-HR/A"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("A2", 1, "RESTATEMENT", "Y"),
        ("A3", 1, "RESTATEMENT", "Y"),  # mismo AN que A2
    ])
    dfs = _mk_dfs(sub, cov, [])
    res = am.apply_amendments(dfs, period="2026-03-31")
    anoms = res["anomalies"]["anomaly_code"].tolist()
    assert am.ANOMALY_AMBIGUOUS_ORDER in anoms


def test_apply_per_cik_period_estructura():
    sub = _mk_sub([("A1", "1", "2026-03-31", "2026-05-15", "13F-HR")])
    cov = _mk_cov([("A1", None, None, None)])
    dfs = _mk_dfs(sub, cov, [("A1", 1, "C1")])
    res = am.apply_amendments(dfs, period="2026-03-31")
    cols = res["per_cik_period"].columns.tolist()
    for c in ["CIK", "report_period", "strategy", "status", "base_accession",
              "applied_accessions", "amendment_count", "filing_count"]:
        assert c in cols


def test_apply_lineage_estructura():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-15", "13F-HR"),
        ("A2", "1", "2026-03-31", "2026-05-19", "13F-HR/A"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("A2", 1, "NEW HOLDINGS", "Y"),
    ])
    dfs = _mk_dfs(sub, cov, [])
    res = am.apply_amendments(dfs, period="2026-03-31")
    cols = res["lineage"].columns.tolist()
    for c in ["CIK", "report_period", "accession", "submission_type",
              "amendment_no", "amendment_type", "operation", "applied", "reason"]:
        assert c in cols


def test_apply_chain_restatement():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-11", "13F-HR"),
        ("A2", "1", "2026-03-31", "2026-05-15", "13F-HR/A"),
        ("A3", "1", "2026-03-31", "2026-05-20", "13F-HR/A"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("A2", 1, "RESTATEMENT", "Y"),
        ("A3", 2, "RESTATEMENT", "Y"),
    ])
    info = [("A1", 1, "C1"), ("A2", 1, "C1"), ("A3", 1, "C1")]
    dfs = _mk_dfs(sub, cov, info)
    res = am.apply_amendments(dfs, period="2026-03-31")
    assert res["applied_accessions"] == {"A3"}


def test_apply_hr_composite():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-08", "13F-HR"),
        ("A2", "1", "2026-03-31", "2026-05-15", "13F-HR/A"),
        ("A3", "1", "2026-03-31", "2026-05-20", "13F-HR/A"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("A2", 1, "RESTATEMENT", "Y"),
        ("A3", 2, "NEW HOLDINGS", "Y"),
    ])
    info = [("A1", 1, "C1"), ("A2", 1, "C1"), ("A3", 2, "C2")]
    dfs = _mk_dfs(sub, cov, info)
    res = am.apply_amendments(dfs, period="2026-03-31")
    assert res["applied_accessions"] == {"A2", "A3"}
    # A1 superado por A2 (restatement)
    lineage = res["lineage"]
    a1_row = lineage[lineage["accession"] == "A1"].iloc[0]
    assert a1_row["applied"] == False


# ---- metricas ----

def test_strategy_counts():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-15", "13F-HR"),
        ("B1", "2", "2026-03-31", "2026-05-15", "13F-NT"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("B1", None, None, None),
    ])
    dfs = _mk_dfs(sub, cov, [])
    res = am.apply_amendments(dfs, period="2026-03-31")
    counts = am.compute_strategy_counts(res["per_cik_period"])
    assert counts[am.STRATEGY_SINGLE_HR] == 1
    assert counts[am.STRATEGY_SINGLE_NOTICE] == 1


def test_status_counts():
    sub = _mk_sub([
        ("A1", "1", "2026-03-31", "2026-05-15", "13F-HR"),
        ("B1", "2", "2026-03-31", "2026-05-15", "13F-NT"),
    ])
    cov = _mk_cov([
        ("A1", None, None, None),
        ("B1", None, None, None),
    ])
    dfs = _mk_dfs(sub, cov, [])
    res = am.apply_amendments(dfs, period="2026-03-31")
    counts = am.compute_status_counts(res["per_cik_period"])
    assert counts.get(am.STATUS_CANONICAL, 0) >= 1
    assert counts.get(am.STATUS_NO_HOLDINGS, 0) >= 1


def test_apply_no_muta_inputs():
    sub = _mk_sub([("A1", "1", "2026-03-31", "2026-05-15", "13F-HR")])
    cov = _mk_cov([("A1", None, None, None)])
    dfs = _mk_dfs(sub, cov, [("A1", 1, "C1")])
    orig_sub_cols = set(sub.columns)
    orig_rows = len(sub)
    am.apply_amendments(dfs, period="2026-03-31")
    assert set(sub.columns) == orig_sub_cols
    assert len(sub) == orig_rows
