# -*- coding: utf-8 -*-
"""D35 (2026-09-30): tests adicionales para scripts/health_check.py."""
import importlib.util
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "health_check",
    str(Path(__file__).resolve().parents[1] / "scripts" / "health_check.py"),
)
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)


# _run_gh
def test_run_gh_ok(monkeypatch):
    fake = MagicMock(returncode=0, stdout="  hola  ")
    monkeypatch.setattr(hc.subprocess, "run", lambda *a, **k: fake)
    assert hc._run_gh(["run", "list"]) == "hola"


def test_run_gh_returncode_no_cero(monkeypatch):
    fake = MagicMock(returncode=1, stdout="")
    monkeypatch.setattr(hc.subprocess, "run", lambda *a, **k: fake)
    assert hc._run_gh(["x"]) is None


def test_run_gh_excepcion(monkeypatch):
    def boom(*a, **k):
        raise OSError("sin gh")
    monkeypatch.setattr(hc.subprocess, "run", boom)
    assert hc._run_gh(["x"]) is None


# _expected_slots
def test_expected_slots_ventana_24h():
    now = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
    cutoff = now - timedelta(hours=24)
    slots = hc._expected_slots(cutoff, now)
    assert len(slots) >= 5
    for s in slots:
        assert cutoff <= s <= now
    assert slots == sorted(slots)


def test_expected_slots_ventana_corta():
    now = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
    cutoff = now - timedelta(minutes=10)
    slots = hc._expected_slots(cutoff, now)
    assert isinstance(slots, list)


# check_cron_slots
def test_check_cron_slots_gh_no_disponible(monkeypatch):
    monkeypatch.setattr(hc, "_run_gh", lambda args: None)
    results = hc.check_cron_slots()
    assert results[0].status == hc.SKIP


def test_check_cron_slots_json_invalido(monkeypatch):
    monkeypatch.setattr(hc, "_run_gh", lambda args: "no-json")
    results = hc.check_cron_slots()
    assert results[0].status == hc.WARN


def test_check_cron_slots_json_lista_vacia(monkeypatch):
    monkeypatch.setattr(hc, "_run_gh", lambda args: "[]")
    results = hc.check_cron_slots()
    assert results[0].status in (hc.FAIL, hc.WARN)


# check_workflows
def test_check_workflows_gh_no_disponible(monkeypatch):
    monkeypatch.setattr(hc, "_run_gh", lambda args: None)
    results = hc.check_workflows()
    assert all(r.status == hc.SKIP for r in results)


def test_check_workflows_ok_con_reciente(monkeypatch):
    now = datetime.now(timezone.utc)
    fake = json.dumps([{
        "createdAt": now.isoformat().replace("+00:00", "Z"),
        "conclusion": "success",
    }])
    monkeypatch.setattr(hc, "_run_gh", lambda args: fake)
    results = hc.check_workflows()
    daily = [r for r in results if "daily_run" in r.name]
    assert daily[0].status == hc.OK


def test_check_workflows_failure_reciente(monkeypatch):
    now = datetime.now(timezone.utc)
    fake = json.dumps([{
        "createdAt": now.isoformat().replace("+00:00", "Z"),
        "conclusion": "failure",
    }])
    monkeypatch.setattr(hc, "_run_gh", lambda args: fake)
    results = hc.check_workflows()
    daily = [r for r in results if "daily_run" in r.name]
    assert daily[0].status == hc.WARN


def test_check_workflows_antiguo_fail(monkeypatch):
    old = datetime.now(timezone.utc) - timedelta(days=30)
    fake = json.dumps([{
        "createdAt": old.isoformat().replace("+00:00", "Z"),
        "conclusion": "success",
    }])
    monkeypatch.setattr(hc, "_run_gh", lambda args: fake)
    results = hc.check_workflows()
    daily = [r for r in results if "daily_run" in r.name]
    assert daily[0].status == hc.FAIL


def test_check_workflows_json_invalido(monkeypatch):
    monkeypatch.setattr(hc, "_run_gh", lambda args: "no-json")
    results = hc.check_workflows()
    assert results[0].status == hc.WARN


# check_13f_cache
def test_check_13f_cache_no_existe(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    results = hc.check_13f_cache()
    assert results[0].status == hc.FAIL


def test_check_13f_cache_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data" / "sec_13f").mkdir(parents=True)
    expected = hc._expected_quarters(date.today(), n=4)
    p = tmp_path / "data" / "sec_13f" / "latest_quarter.txt"
    p.write_text(expected[0], encoding="utf-8")
    results = hc.check_13f_cache()
    assert results[0].status == hc.OK


def test_check_13f_cache_warn(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data" / "sec_13f").mkdir(parents=True)
    p = tmp_path / "data" / "sec_13f" / "latest_quarter.txt"
    p.write_text("2020Q1", encoding="utf-8")
    results = hc.check_13f_cache()
    assert results[0].status == hc.WARN


# check_yahoo_revision
def test_check_yahoo_revision_manifest_ausente(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    results = hc.check_yahoo_revision()
    assert results[0].status == hc.SKIP


def test_check_yahoo_revision_manifest_invalido(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    d = tmp_path / "data"
    d.mkdir()
    (d / "market_data.parquet.manifest.json").write_text("no-json", encoding="utf-8")
    results = hc.check_yahoo_revision()
    assert results[0].status == hc.WARN


def test_check_yahoo_revision_manifest_no_en_head(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    d = tmp_path / "data"
    d.mkdir()
    manifest = {"artifact": {"sha256": "a"}, "quality": {"last_date": "2026-09-30"}}
    (d / "market_data.parquet.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def fake_run(args, **kw):
        m = MagicMock()
        m.returncode = 128
        m.stdout = ""
        return m
    monkeypatch.setattr(hc.subprocess, "run", fake_run)
    results = hc.check_yahoo_revision()
    assert results[0].status == hc.SKIP


# check_eu_usa_pattern
def _mk_df(data):
    idx = pd.date_range("2024-01-01", periods=len(next(iter(data.values()))), freq="B")
    cols = pd.MultiIndex.from_tuples([("Close", t) for t in data])
    df = pd.DataFrame(data, index=idx)
    df.columns = cols
    return df


def test_check_eu_usa_pattern_sin_datos():
    df = pd.DataFrame()
    results = hc.check_eu_usa_pattern(df)
    assert results[0].status == hc.SKIP


def test_check_eu_usa_pattern_normal():
    n = 10
    data = {}
    for i in range(10):
        v = [100.0] * n if i < 6 else [float("nan")] * n
        data["XX" + str(i) + ".PA"] = v
    for i in range(10):
        data["AA" + str(i)] = [100.0] * n
    df = _mk_df(data)
    results = hc.check_eu_usa_pattern(df)
    assert results[0].status in (hc.OK, hc.WARN)


# check_iae_section
REPORTE_IAE = "## Acumulacion Institucional (13F)"

def test_check_iae_section_reporte_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    r = tmp_path / "outputs" / "report"
    r.mkdir(parents=True)
    (r / "reporte_diario.md").write_text(
        REPORTE_IAE + chr(10) + "Contenido OK" + chr(10), encoding="utf-8")
    results = hc.check_iae_section(is_ci=False)
    assert results[0].status == hc.OK


def test_check_iae_section_reporte_stale_official_list(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    r = tmp_path / "outputs" / "report"
    r.mkdir(parents=True)
    (r / "reporte_diario.md").write_text(
        REPORTE_IAE + chr(10) + "STALE" + chr(10) + "Official List" + chr(10),
        encoding="utf-8")
    results = hc.check_iae_section(is_ci=False)
    assert results[0].status == hc.OK


def test_check_iae_section_reporte_stale_desconocido(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    r = tmp_path / "outputs" / "report"
    r.mkdir(parents=True)
    (r / "reporte_diario.md").write_text(
        REPORTE_IAE + chr(10) + "STALE sin razon" + chr(10), encoding="utf-8")
    results = hc.check_iae_section(is_ci=False)
    assert results[0].status == hc.WARN


# _render_markdown
def test_render_markdown_simple():
    results = [
        hc.Result("a", hc.OK, "ok"),
        hc.Result("b", hc.WARN, "warn"),
        hc.Result("c", hc.FAIL, "fail"),
    ]
    now = datetime(2026, 9, 30, 10, 0, tzinfo=timezone.utc)
    md = hc._render_markdown(results, now)
    assert "Health Check - 2026-09-30 10:00 UTC" in md
    assert "[OK]" in md
    assert "[WARN]" in md
    assert "[FAIL]" in md
    assert "Resumen:" in md


def test_render_markdown_resumen_counts():
    results = [hc.Result("a", hc.OK, ""), hc.Result("b", hc.OK, "")]
    now = datetime(2026, 9, 30, 10, 0, tzinfo=timezone.utc)
    md = hc._render_markdown(results, now)
    assert "2 OK" in md


# _ensure_label / _find_open_issue
def test_ensure_label_invoca_subprocess(monkeypatch):
    """_ensure_label usa subprocess.run directo, no _run_gh."""
    calls = []
    def fake_run(args, **kw):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(hc.subprocess, "run", fake_run)
    hc._ensure_label()
    assert len(calls) == 1
    assert "gh" in calls[0]
    assert "label" in calls[0]
    assert "create" in calls[0]


def test_find_open_issue_sin_issue(monkeypatch):
    """Lista vacia -> None."""
    monkeypatch.setattr(hc, "_run_gh", lambda args: "[]")
    assert hc._find_open_issue() is None


def test_find_open_issue_numero(monkeypatch):
    """Lista con 1 issue -> numero."""
    monkeypatch.setattr(hc, "_run_gh",
                        lambda args: '[{"number": 42}]')
    assert hc._find_open_issue() == 42


def test_find_open_issue_gh_none(monkeypatch):
    monkeypatch.setattr(hc, "_run_gh", lambda args: None)
    assert hc._find_open_issue() is None


def test_find_open_issue_json_invalido(monkeypatch):
    monkeypatch.setattr(hc, "_run_gh", lambda args: "no-json")
    assert hc._find_open_issue() is None
