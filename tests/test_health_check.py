# -*- coding: utf-8 -*-
"""Tests de health_check (funciones puras, sin gh)."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

# Importar el script
spec = importlib.util.spec_from_file_location(
    "health_check",
    str(Path(__file__).resolve().parents[1] / "scripts" / "health_check.py"),
)
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)


# ---------- Helpers ----------
def _make_df(tickers, n_rows=10, start="2026-09-01"):
    idx = pd.date_range(start, periods=n_rows, freq="B")
    data = {}
    for t in tickers:
        data[("Close", t)] = [100.0] * n_rows
    return pd.DataFrame(data, index=idx)


# ---------- Check D: coverage ----------
def test_coverage_last_5_todo_ok():
    df = _make_df(["AAPL", "MSFT", "GOOG"], n_rows=10)
    results = hc.check_coverage_last_5(df)
    assert all(r.status == hc.OK for r in results)


def test_coverage_last_5_ultima_baja_fail():
    df = _make_df(["AAPL", "MSFT", "GOOG"], n_rows=10)
    # Ultima fila: 1 de 3 validos = 33% -> FAIL (<50%)
    df.iloc[-1, 0] = None
    df.iloc[-1, 1] = None
    results = hc.check_coverage_last_5(df)
    last = [r for r in results if r.name == "coverage:last"][0]
    assert last.status == hc.FAIL


def test_coverage_last_5_hist_warn():
    df = _make_df(["AAPL", "MSFT", "GOOG"], n_rows=10)
    # Penultima fila: 1 de 3 validos = 33% -> WARN hist
    df.iloc[-2, 0] = None
    df.iloc[-2, 1] = None
    results = hc.check_coverage_last_5(df)
    hist = [r for r in results if r.name == "coverage:hist"][0]
    assert hist.status == hc.WARN


def test_coverage_sin_columnas_close_fail():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    results = hc.check_coverage_last_5(df)
    assert results[0].status == hc.FAIL


# ---------- Check E: fechas no bursatiles ----------
def test_fechas_no_bursatiles_todas_bursatiles():
    df = _make_df(["AAPL"], n_rows=5, start="2026-09-14")
    results = hc.check_non_market_days(df)
    # 14-18 sep son L,M,X,J,V - todas bursatiles
    assert results[0].status == hc.OK


def test_fechas_no_bursatiles_con_finde():
    # Indice con sabado incluido
    idx = pd.to_datetime(["2026-09-18", "2026-09-19", "2026-09-21"])
    df = pd.DataFrame({("Close", "AAPL"): [100.0] * 3}, index=idx)
    results = hc.check_non_market_days(df)
    assert results[0].status == hc.WARN


# ---------- Check C: manifest ----------
def test_manifest_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    d = tmp_path / "data"
    d.mkdir()
    m = {"quality": {"status": "VALID", "last_date": "2026-09-23"}}
    (d / "foo.parquet.manifest.json").write_text(json.dumps(m))
    results = hc.check_manifest("foo")
    assert results[0].status == hc.OK


def test_manifest_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    d = tmp_path / "data"
    d.mkdir()
    m = {"quality": {"status": "INVALID", "last_date": "2026-09-25",
                     "expected_session": "2026-09-24"}}
    (d / "foo.parquet.manifest.json").write_text(json.dumps(m))
    results = hc.check_manifest("foo")
    assert results[0].status == hc.FAIL


def test_manifest_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    results = hc.check_manifest("noexiste")
    assert results[0].status == hc.FAIL


def test_manifest_valid_with_missing_warn(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    d = tmp_path / "data"
    d.mkdir()
    m = {"quality": {"status": "VALID_WITH_MISSING", "close_nan_last": 5}}
    (d / "foo.parquet.manifest.json").write_text(json.dumps(m))
    results = hc.check_manifest("foo")
    assert results[0].status == hc.WARN


# ---------- Check B: quarters ----------
def test_expected_quarters():
    from datetime import date
    qs = hc._expected_quarters(date(2026, 9, 24), n=4)
    # 2026-09-24 esta en Q3 -> Q-1=Q2, Q-2=Q1, Q-3=2025Q4, Q-4=2025Q3
    assert qs[0] == "2026Q2"
    assert "2025Q4" in qs


# ---------- Overall ----------
def test_overall_fail_domina():
    results = [hc.Result("a", hc.OK, ""), hc.Result("b", hc.FAIL, "")]
    assert hc._overall_status(results) == hc.FAIL


def test_overall_warn_si_no_fail():
    results = [hc.Result("a", hc.OK, ""), hc.Result("b", hc.WARN, "")]
    assert hc._overall_status(results) == hc.WARN


def test_overall_ok():
    results = [hc.Result("a", hc.OK, ""), hc.Result("b", hc.OK, "")]
    assert hc._overall_status(results) == hc.OK

# ---------- _update_issue: gh falla -> no miente ----------
def test_update_issue_close_gh_falla_no_dice_cerrada(monkeypatch, capsys):
    monkeypatch.setattr(hc, "_ensure_label", lambda: None)
    monkeypatch.setattr(hc, "_find_open_issue", lambda: 42)
    monkeypatch.setattr(hc, "_run_gh", lambda args: None)
    hc._update_issue(hc.OK, "body")
    out = capsys.readouterr().out
    assert "cerrada" not in out
    assert "WARN" in out
    assert "#42" in out


def test_update_issue_comment_gh_falla_no_dice_actualizada(monkeypatch, capsys):
    monkeypatch.setattr(hc, "_ensure_label", lambda: None)
    monkeypatch.setattr(hc, "_find_open_issue", lambda: 42)
    monkeypatch.setattr(hc, "_run_gh", lambda args: None)
    hc._update_issue(hc.WARN, "body")
    out = capsys.readouterr().out
    assert "actualizada" not in out
    assert "WARN" in out
    assert "#42" in out


def test_update_issue_close_gh_ok_dice_cerrada(monkeypatch, capsys):
    monkeypatch.setattr(hc, "_ensure_label", lambda: None)
    monkeypatch.setattr(hc, "_find_open_issue", lambda: 42)
    monkeypatch.setattr(hc, "_run_gh", lambda args: "ok")
    hc._update_issue(hc.OK, "body")
    out = capsys.readouterr().out
    assert "cerrada" in out
    assert "WARN" not in out


# ---------- Check H: cron_slots (umbral) ----------
def test_cron_slots_status_ok():
    assert hc._cron_slots_status(0) == hc.OK


def test_cron_slots_status_1_missing_warn():
    assert hc._cron_slots_status(1) == hc.WARN


def test_cron_slots_status_2_missing_warn():
    assert hc._cron_slots_status(2) == hc.WARN


def test_cron_slots_status_3_missing_fail():
    assert hc._cron_slots_status(3) == hc.FAIL


def test_cron_slots_status_5_missing_fail():
    assert hc._cron_slots_status(5) == hc.FAIL


# ---------- Check A: workflows trimestrales ----------
def test_workflows_trimestral_sin_runs_skip(monkeypatch):
    monkeypatch.setattr(hc, "_run_gh", lambda args: "[]")
    results = hc.check_workflows()
    trim = [r for r in results if "update_sec_13f" in r.name]
    assert trim and trim[0].status == hc.SKIP


def test_workflows_daily_sin_runs_warn(monkeypatch):
    monkeypatch.setattr(hc, "_run_gh", lambda args: "[]")
    results = hc.check_workflows()
    daily = [r for r in results if "daily_run" in r.name]
    assert daily and daily[0].status == hc.WARN


# ---------- Check G: iae_section ----------
def test_iae_section_sin_reporte_en_ci_skip(monkeypatch, tmp_path):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    results = hc.check_iae_section(is_ci=True)
    assert results[0].status == hc.SKIP


def test_iae_section_sin_reporte_en_local_warn(monkeypatch, tmp_path):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    results = hc.check_iae_section(is_ci=False)
    assert results[0].status == hc.WARN


# ---------- run_all_checks: parquet en CI vs local ----------
def test_run_all_checks_parquet_skip_en_ci(monkeypatch, tmp_path):
    monkeypatch.setattr(hc, "IS_CI", True)
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hc, "check_workflows", lambda: [])
    monkeypatch.setattr(hc, "check_cron_slots", lambda: [])
    monkeypatch.setattr(hc, "check_13f_cache", lambda: [])
    monkeypatch.setattr(hc, "check_manifest", lambda name: [])
    monkeypatch.setattr(hc, "check_iae_section", lambda is_ci=False: [])
    results = hc.run_all_checks()
    parquet = [r for r in results if r.name == "parquet"]
    assert len(parquet) == 1
    assert parquet[0].status == hc.SKIP


def test_run_all_checks_parquet_fail_en_local(monkeypatch, tmp_path):
    monkeypatch.setattr(hc, "IS_CI", False)
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hc, "check_workflows", lambda: [])
    monkeypatch.setattr(hc, "check_cron_slots", lambda: [])
    monkeypatch.setattr(hc, "check_13f_cache", lambda: [])
    monkeypatch.setattr(hc, "check_manifest", lambda name: [])
    monkeypatch.setattr(hc, "check_iae_section", lambda is_ci=False: [])
    results = hc.run_all_checks()
    parquet = [r for r in results if r.name == "parquet"]
    assert len(parquet) == 1
    assert parquet[0].status == hc.FAIL
