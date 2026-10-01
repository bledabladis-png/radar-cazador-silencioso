# -*- coding: utf-8 -*-
"""D37 (2026-09-30): tests para 3 scripts de orquestacion.

- scripts/pipeline_gate.py: 73% -> cubre _read_manifest,
  _sha256_file, _manifest_satisfies, _probe_panel_once (mock),
  _probe_panel (retry), resolve_slot_flags, evaluate,
  _write_github_output, resolve_target_session.
- scripts/guard_coverage.py: 76% -> cubre _load_manifest,
  _sha256_file, _all_markets_valid, _check_manifest.
- scripts/issue_manager.py: 77% -> cubre validate_target_session,
  decide_action, _run_gh, _find_issue_by_title, _ensure_label,
  execute_action (CLOSE / ENSURE_FAILURE).
"""
import importlib.util
import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name,
        str(Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")),
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pg = _load("pipeline_gate")
gc = _load("guard_coverage")
im = _load("issue_manager")


# ============================================================
# pipeline_gate._read_manifest
# ============================================================
def test_pg_read_manifest_no_existe(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "PROJECT_ROOT", tmp_path)
    assert pg._read_manifest("nope.json") is None


def test_pg_read_manifest_json_invalido(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "PROJECT_ROOT", tmp_path)
    (tmp_path / "bad.json").write_text("no-json", encoding="utf-8")
    assert pg._read_manifest("bad.json") is None


def test_pg_read_manifest_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "PROJECT_ROOT", tmp_path)
    (tmp_path / "ok.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert pg._read_manifest("ok.json") == {"a": 1}


# ============================================================
# pipeline_gate._sha256_file
# ============================================================
def test_pg_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    # sha256("hello")
    expected = ("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
    assert pg._sha256_file(str(f)) == expected


# ============================================================
# pipeline_gate._manifest_satisfies
# ============================================================
def _manifest_ok(tmp_path, coverage=0.99, expected="2026-09-29",
                  sha=None):
    if sha is None:
        sha = "a" * 64
    return {
        "quality": {
            "expected_session": expected,
            "coverage_pct_last": coverage,
        },
        "artifact": {"sha256": sha},
    }


def test_pg_manifest_satisfies_no_dict():
    assert not pg._manifest_satisfies(None, "2026-09-29")
    assert not pg._manifest_satisfies("string", "2026-09-29")


def test_pg_manifest_satisfies_quality_ausente():
    assert not pg._manifest_satisfies({"artifact": {}}, "2026-09-29")


def test_pg_manifest_satisfies_session_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "PROJECT_ROOT", tmp_path)
    m = _manifest_ok(tmp_path, expected="2026-01-01")
    assert not pg._manifest_satisfies(m, "2026-09-29")


def test_pg_manifest_satisfies_coverage_insuficiente(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "PROJECT_ROOT", tmp_path)
    m = _manifest_ok(tmp_path, coverage=0.5)
    assert not pg._manifest_satisfies(m, "2026-09-29")


def test_pg_manifest_satisfies_coverage_no_numerica(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "PROJECT_ROOT", tmp_path)
    m = _manifest_ok(tmp_path)
    m["quality"]["coverage_pct_last"] = "high"
    assert not pg._manifest_satisfies(m, "2026-09-29")


def test_pg_manifest_satisfies_sha_ausente(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "PROJECT_ROOT", tmp_path)
    m = _manifest_ok(tmp_path)
    m["artifact"] = {}
    assert not pg._manifest_satisfies(m, "2026-09-29")


def test_pg_manifest_satisfies_parquet_ausente(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pg, "MANIFEST_PATH",
                        "data/stock_prices.parquet.manifest.json")
    m = _manifest_ok(tmp_path)
    assert not pg._manifest_satisfies(m, "2026-09-29")


def test_pg_manifest_satisfies_sha_match(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pg, "MANIFEST_PATH",
                        "data/stock_prices.parquet.manifest.json")
    (tmp_path / "data").mkdir()
    p = tmp_path / "data" / "stock_prices.parquet"
    p.write_bytes(b"contenido")
    import hashlib
    real = hashlib.sha256(b"contenido").hexdigest()
    m = _manifest_ok(tmp_path, sha=real)
    assert pg._manifest_satisfies(m, "2026-09-29")


def test_pg_manifest_satisfies_sha_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pg, "MANIFEST_PATH",
                        "data/stock_prices.parquet.manifest.json")
    (tmp_path / "data").mkdir()
    p = tmp_path / "data" / "stock_prices.parquet"
    p.write_bytes(b"contenido")
    m = _manifest_ok(tmp_path, sha="b" * 64)
    assert not pg._manifest_satisfies(m, "2026-09-29")


# ============================================================
# pipeline_gate._probe_panel_once
# ============================================================
def test_pg_probe_panel_once_yf_excepcion(monkeypatch):
    def boom(*a, **k):
        raise OSError("sin red")
    monkeypatch.setattr(pg.yf, "download", boom)
    cov, err = pg._probe_panel_once("2026-09-29", ("AAPL", "MSFT"))
    assert cov == 0.0
    assert "probe failed" in err


def test_pg_probe_panel_once_data_vacia(monkeypatch):
    monkeypatch.setattr(pg.yf, "download",
                        lambda *a, **k: pd.DataFrame())
    cov, err = pg._probe_panel_once("2026-09-29", ("AAPL", "MSFT"))
    assert cov == 0.0
    assert err is None


def test_pg_probe_panel_once_sin_close(monkeypatch):
    """DataFrame sin columna Close."""
    df = pd.DataFrame({"Open": [1, 2]}, index=pd.date_range("2026-09-29", periods=2))
    monkeypatch.setattr(pg.yf, "download", lambda *a, **k: df)
    cov, err = pg._probe_panel_once("2026-09-29", ("AAPL",))
    assert cov == 0.0
    assert err is None


def test_pg_probe_panel_once_close_ok(monkeypatch):
    """MultiIndex Close con target en index."""
    ts = pd.Timestamp("2026-09-29")
    df = pd.DataFrame(
        {("Close", "AAPL"): [100.0], ("Close", "MSFT"): [200.0]},
        index=[ts],
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    monkeypatch.setattr(pg.yf, "download", lambda *a, **k: df)
    cov, err = pg._probe_panel_once("2026-09-29", ("AAPL", "MSFT"))
    assert cov == 1.0
    assert err is None


def test_pg_probe_panel_once_target_no_en_index(monkeypatch):
    df = pd.DataFrame(
        {("Close", "AAPL"): [100.0]},
        index=[pd.Timestamp("2026-01-01")],
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    monkeypatch.setattr(pg.yf, "download", lambda *a, **k: df)
    cov, err = pg._probe_panel_once("2026-09-29", ("AAPL",))
    assert cov == 0.0
    assert err is None


# ============================================================
# pipeline_gate._probe_panel: retry
# ============================================================
def test_pg_probe_panel_exito_primer_intento(monkeypatch):
    monkeypatch.setattr(pg, "_probe_panel_once",
                        lambda s, t: (0.99, None))
    cov, err = pg._probe_panel("2026-09-29")
    assert cov == 0.99
    assert err is None


def test_pg_probe_panel_error_tras_retries(monkeypatch):
    monkeypatch.setattr(pg, "_probe_panel_once",
                        lambda s, t: (0.0, "boom"))
    monkeypatch.setattr(pg.time, "sleep", lambda s: None)
    cov, err = pg._probe_panel("2026-09-29")
    assert cov == 0.0
    assert err == "boom"


def test_pg_probe_panel_error_luego_ok(monkeypatch):
    calls = {"n": 0}
    def fake(s, t):
        calls["n"] += 1
        if calls["n"] == 1:
            return 0.0, "boom"
        return 0.99, None
    monkeypatch.setattr(pg, "_probe_panel_once", fake)
    monkeypatch.setattr(pg.time, "sleep", lambda s: None)
    cov, err = pg._probe_panel("2026-09-29")
    assert cov == 0.99
    assert err is None


# ============================================================
# pipeline_gate.resolve_slot_flags
# ============================================================
def test_pg_resolve_slot_flags_vacio():
    assert pg.resolve_slot_flags("") == (False, False)


def test_pg_resolve_slot_flags_manual():
    assert pg.resolve_slot_flags("manual") == (False, False)


def test_pg_resolve_slot_flags_desconocido():
    assert pg.resolve_slot_flags("0 0 1 1 1") == (False, False)


def test_pg_resolve_slot_flags_conocido_no_ultimo():
    assert pg.resolve_slot_flags("17 3 * * *") == (True, False)


def test_pg_resolve_slot_flags_ultimo():
    assert pg.resolve_slot_flags("17 11 * * *") == (True, True)


# ============================================================
# pipeline_gate.evaluate
# ============================================================
def test_pg_evaluate_current(monkeypatch, tmp_path):
    # Contrato v1 (2026-10-01): CURRENT requiere receipt valido.
    # _manifest_satisfies dejo de ser la fuente de CURRENT.
    monkeypatch.setattr(pg, "find_completion_receipt",
                        lambda t: {"run_id": 99999})
    r = pg.evaluate("2026-09-29")
    assert r["state"] == "CURRENT"
    assert r["should_run"] is False


def test_pg_evaluate_error(monkeypatch):
    monkeypatch.setattr(pg, "_read_manifest", lambda p: None)
    monkeypatch.setattr(pg, "_manifest_satisfies", lambda m, t: False)
    monkeypatch.setattr(pg, "_probe_panel",
                        lambda s: (0.0, "red"))
    r = pg.evaluate("2026-09-29")
    assert r["state"] == "ERROR"
    assert r["should_run"] is False


def test_pg_evaluate_ready(monkeypatch):
    monkeypatch.setattr(pg, "_read_manifest", lambda p: None)
    monkeypatch.setattr(pg, "_manifest_satisfies", lambda m, t: False)
    monkeypatch.setattr(pg, "_probe_panel", lambda s: (0.99, None))
    r = pg.evaluate("2026-09-29")
    assert r["state"] == "READY"
    assert r["should_run"] is True


def test_pg_evaluate_not_ready(monkeypatch):
    monkeypatch.setattr(pg, "_read_manifest", lambda p: None)
    monkeypatch.setattr(pg, "_manifest_satisfies", lambda m, t: False)
    monkeypatch.setattr(pg, "_probe_panel", lambda s: (0.5, None))
    r = pg.evaluate("2026-09-29")
    assert r["state"] == "NOT_READY"


# ============================================================
# pipeline_gate._write_github_output
# ============================================================
def test_pg_write_github_output_sin_env(monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    # No debe crashear
    pg._write_github_output({"state": "OK", "should_run": True,
                              "expected_session": "2026-09-29",
                              "reason": "x"})


def test_pg_write_github_output_con_env(tmp_path, monkeypatch):
    out = tmp_path / "gh.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    r = {
        "state": "READY",
        "should_run": True,
        "expected_session": "2026-09-29",
        "reason": "coverage ok",
        "is_known_slot": True,
        "is_last_slot": True,
    }
    pg._write_github_output(r)
    text = out.read_text(encoding="utf-8")
    assert "state=READY" in text
    assert "should_run=true" in text
    assert "expected_session=2026-09-29" in text
    assert "is_last_slot=true" in text


# ============================================================
# pipeline_gate.resolve_target_session
# ============================================================
def test_pg_resolve_target_session_con_now():
    now = datetime(2026, 9, 30, 12, 0)
    out = pg.resolve_target_session(now)
    assert isinstance(out, str)
    # Formato ISO date
    date.fromisoformat(out)


def test_pg_resolve_target_session_default():
    out = pg.resolve_target_session()
    date.fromisoformat(out)


# ============================================================
# guard_coverage._load_manifest
# ============================================================
def test_gc_load_manifest_missing(tmp_path):
    d, err = gc._load_manifest(str(tmp_path / "nope.json"))
    assert d is None
    assert "missing" in err


def test_gc_load_manifest_json_invalido(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("no-json", encoding="utf-8")
    d, err = gc._load_manifest(str(p))
    assert d is None
    assert "JSON" in err or "invalid" in err


def test_gc_load_manifest_root_no_dict(tmp_path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    d, err = gc._load_manifest(str(p))
    assert d is None
    assert "dict" in err


def test_gc_load_manifest_ok(tmp_path):
    p = tmp_path / "ok.json"
    p.write_text(json.dumps({"a": 1}), encoding="utf-8")
    d, err = gc._load_manifest(str(p))
    assert d == {"a": 1}
    assert err is None


# ============================================================
# guard_coverage._sha256_file
# ============================================================
def test_gc_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    expected = ("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
    assert gc._sha256_file(str(f)) == expected


# ============================================================
# guard_coverage._all_markets_valid
# ============================================================
def test_gc_all_markets_valid_vacio():
    assert not gc._all_markets_valid({}, 0.95)
    assert not gc._all_markets_valid(None, 0.95)


def test_gc_all_markets_valid_todos_skip():
    bm = {"BME": {"n": 0, "status": "SKIP"},
          "XETRA": {"n": 0, "status": "SKIP"}}
    assert not gc._all_markets_valid(bm, 0.95)


def test_gc_all_markets_valid_ok():
    bm = {"US": {"n": 100, "status": "VALID", "coverage_at_session": 0.99}}
    assert gc._all_markets_valid(bm, 0.95)


def test_gc_all_markets_valid_invalid_falla():
    bm = {"US": {"n": 100, "status": "INVALID", "coverage_at_session": 0.99}}
    assert not gc._all_markets_valid(bm, 0.95)


def test_gc_all_markets_valid_coverage_baja():
    bm = {"US": {"n": 100, "status": "VALID", "coverage_at_session": 0.5}}
    assert not gc._all_markets_valid(bm, 0.95)


def test_gc_all_markets_valid_pending():
    """PENDING se ignora; si todo es PENDING, False."""
    bm = {"BME": {"n": 100, "status": "PENDING"}}
    assert not gc._all_markets_valid(bm, 0.95)


def test_gc_all_markets_valid_pending_y_valid():
    """PENDING + VALID >= threshold -> True."""
    bm = {"BME": {"n": 100, "status": "PENDING"},
          "US": {"n": 100, "status": "VALID", "coverage_at_session": 0.99}}
    assert gc._all_markets_valid(bm, 0.95)


def test_gc_all_markets_valid_info_no_dict():
    bm = {"US": "no-dict"}
    assert not gc._all_markets_valid(bm, 0.95)


# ============================================================
# guard_coverage._check_manifest
# ============================================================
def test_gc_check_manifest_missing(tmp_path):
    reasons = gc._check_manifest(str(tmp_path / "nope.json"), 0.95)
    assert reasons
    assert "missing" in reasons[0]


def test_gc_check_manifest_quality_ausente(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    reasons = gc._check_manifest(str(p), 0.95)
    assert any("quality" in r for r in reasons)


def test_gc_check_manifest_full_ok(tmp_path):
    """Manifest completo con parquet, sha match, cobertura, fechas."""
    import hashlib
    (tmp_path / "data").mkdir()
    parquet = tmp_path / "data" / "x.parquet"
    parquet.write_bytes(b"contenido")
    real_sha = hashlib.sha256(b"contenido").hexdigest()
    m = {
        "artifact": {"sha256": real_sha},
        "quality": {
            "status": "VALID",
            "coverage_pct_last": 0.99,
            "last_date": "2026-09-29",
            "expected_session": "2026-09-29",
        },
    }
    mp = tmp_path / "data" / "x.parquet.manifest.json"
    mp.write_text(json.dumps(m), encoding="utf-8")
    reasons = gc._check_manifest(str(mp), 0.95)
    assert reasons == []


def test_gc_check_manifest_sha_mismatch(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.parquet").write_bytes(b"real")
    m = {
        "artifact": {"sha256": "a" * 64},
        "quality": {
            "status": "VALID",
            "coverage_pct_last": 0.99,
            "last_date": "2026-09-29",
            "expected_session": "2026-09-29",
        },
    }
    mp = tmp_path / "data" / "x.parquet.manifest.json"
    mp.write_text(json.dumps(m), encoding="utf-8")
    reasons = gc._check_manifest(str(mp), 0.95)
    assert any("sha256 mismatch" in r for r in reasons)


def test_gc_check_manifest_status_invalid(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.parquet").write_bytes(b"real")
    m = {
        "artifact": {"sha256": "a" * 64},
        "quality": {
            "status": "INVALID",
            "coverage_pct_last": 0.99,
            "last_date": "2026-09-29",
            "expected_session": "2026-09-29",
        },
    }
    mp = tmp_path / "data" / "x.parquet.manifest.json"
    mp.write_text(json.dumps(m), encoding="utf-8")
    reasons = gc._check_manifest(str(mp), 0.95)
    assert any("INVALID" in r for r in reasons)


def test_gc_check_manifest_last_mayor_expected(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.parquet").write_bytes(b"real")
    import hashlib
    real_sha = hashlib.sha256(b"real").hexdigest()
    m = {
        "artifact": {"sha256": real_sha},
        "quality": {
            "status": "VALID",
            "coverage_pct_last": 0.99,
            "last_date": "2026-09-30",
            "expected_session": "2026-09-29",
        },
    }
    mp = tmp_path / "data" / "x.parquet.manifest.json"
    mp.write_text(json.dumps(m), encoding="utf-8")
    reasons = gc._check_manifest(str(mp), 0.95)
    assert any("last_date" in r and "expected" in r for r in reasons)


def test_gc_check_manifest_coverage_baja_sin_exencion(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.parquet").write_bytes(b"real")
    import hashlib
    real_sha = hashlib.sha256(b"real").hexdigest()
    m = {
        "artifact": {"sha256": real_sha},
        "quality": {
            "status": "VALID",
            "coverage_pct_last": 0.5,
            "last_date": "2026-09-29",
            "expected_session": "2026-09-29",
        },
    }
    mp = tmp_path / "data" / "x.parquet.manifest.json"
    mp.write_text(json.dumps(m), encoding="utf-8")
    reasons = gc._check_manifest(str(mp), 0.95)
    assert any("coverage_pct_last" in r for r in reasons)


# ============================================================
# issue_manager.validate_target_session
# ============================================================
def test_im_validate_target_session_vacio():
    assert not im.validate_target_session("")
    assert not im.validate_target_session(None)


def test_im_validate_target_session_invalido():
    assert not im.validate_target_session("no-es-fecha")
    assert not im.validate_target_session("2026-13-01")


def test_im_validate_target_session_ok():
    assert im.validate_target_session("2026-09-29")


# ============================================================
# issue_manager.decide_action
# ============================================================
def test_im_decide_action_estado_invalido():
    assert im.decide_action("BOGUS", "success", False, "schedule") == im.ACTION_NOOP


def test_im_decide_action_evento_invalido():
    assert im.decide_action("READY", "success", False, "push") == im.ACTION_NOOP


def test_im_decide_action_current_close():
    assert im.decide_action("CURRENT", "skipped", False, "schedule") == im.ACTION_CLOSE


def test_im_decide_action_ready_success_close():
    assert im.decide_action("READY", "success", False, "schedule") == im.ACTION_CLOSE


def test_im_decide_action_ready_failure_noop():
    assert im.decide_action("READY", "failure", False, "schedule") == im.ACTION_NOOP


def test_im_decide_action_ready_skipped_noop():
    assert im.decide_action("READY", "skipped", False, "schedule") == im.ACTION_NOOP


def test_im_decide_action_not_ready_last_slot_schedule():
    assert im.decide_action(
        "NOT_READY", "skipped", True, "schedule") == im.ACTION_ENSURE_FAILURE


def test_im_decide_action_not_ready_no_last_noop():
    assert im.decide_action(
        "NOT_READY", "skipped", False, "schedule") == im.ACTION_NOOP


def test_im_decide_action_error_last_slot_schedule():
    assert im.decide_action(
        "ERROR", "skipped", True, "schedule") == im.ACTION_ENSURE_FAILURE


def test_im_decide_action_error_non_last_noop():
    assert im.decide_action(
        "ERROR", "skipped", False, "schedule") == im.ACTION_NOOP


def test_im_decide_action_dispatch_not_ready_no_ensure():
    """workflow_dispatch nunca ENSURE_FAILURE."""
    assert im.decide_action(
        "NOT_READY", "skipped", True, "workflow_dispatch") == im.ACTION_NOOP


def test_im_decide_action_dispatch_current_close():
    assert im.decide_action(
        "CURRENT", "skipped", True, "workflow_dispatch") == im.ACTION_CLOSE


def test_im_decide_action_dispatch_ready_success_close():
    assert im.decide_action(
        "READY", "success", True, "workflow_dispatch") == im.ACTION_CLOSE


# ============================================================
# issue_manager._run_gh
# ============================================================
def test_im_run_gh_ok(monkeypatch):
    fake = MagicMock(returncode=0, stdout="hi")
    monkeypatch.setattr(im.subprocess, "run", lambda *a, **k: fake)
    assert im._run_gh(["x"]) == "hi"


def test_im_run_gh_error(monkeypatch):
    fake = MagicMock(returncode=1, stdout="")
    monkeypatch.setattr(im.subprocess, "run", lambda *a, **k: fake)
    assert im._run_gh(["x"]) is None


def test_im_run_gh_excepcion(monkeypatch):
    def boom(*a, **k):
        raise OSError("no gh")
    monkeypatch.setattr(im.subprocess, "run", boom)
    assert im._run_gh(["x"]) is None


# ============================================================
# issue_manager._find_issue_by_title
# ============================================================
def test_im_find_issue_by_title_gh_none(monkeypatch):
    monkeypatch.setattr(im, "_run_gh", lambda a: None)
    assert im._find_issue_by_title("2026-09-29") is None


def test_im_find_issue_by_title_json_invalido(monkeypatch):
    monkeypatch.setattr(im, "_run_gh", lambda a: "no-json")
    assert im._find_issue_by_title("2026-09-29") is None


def test_im_find_issue_by_title_match(monkeypatch):
    title = im.TITLE_TEMPLATE.format(session="2026-09-29")
    payload = json.dumps([
        {"number": 10, "title": "otro"},
        {"number": 42, "title": title},
    ])
    monkeypatch.setattr(im, "_run_gh", lambda a: payload)
    assert im._find_issue_by_title("2026-09-29") == 42


def test_im_find_issue_by_title_sin_match(monkeypatch):
    payload = json.dumps([{"number": 10, "title": "otro"}])
    monkeypatch.setattr(im, "_run_gh", lambda a: payload)
    assert im._find_issue_by_title("2026-09-29") is None


# ============================================================
# issue_manager._ensure_label
# ============================================================
def test_im_ensure_label(monkeypatch):
    calls = []
    def fake(args, **k):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(im.subprocess, "run", fake)
    im._ensure_label()
    assert len(calls) == 1
    assert "gh" in calls[0]
    assert "label" in calls[0]
    assert "create" in calls[0]


# ============================================================
# issue_manager.execute_action
# ============================================================
def test_im_execute_action_noop():
    # No debe hacer nada
    im.execute_action(im.ACTION_NOOP, target_session="2026-09-29",
                       reason="x", run_id="1")


def test_im_execute_action_target_invalido(capsys):
    im.execute_action(im.ACTION_CLOSE, target_session="bogus",
                       reason="x", run_id="1")
    out = capsys.readouterr().out
    assert "invalido" in out


def test_im_execute_action_close_sin_issue(monkeypatch, capsys):
    monkeypatch.setattr(im, "_find_issue_by_title", lambda s: None)
    im.execute_action(im.ACTION_CLOSE, target_session="2026-09-29",
                       reason="x", run_id="1")
    out = capsys.readouterr().out
    assert "no hay issue" in out


def test_im_execute_action_close_con_issue(monkeypatch, capsys):
    monkeypatch.setattr(im, "_find_issue_by_title", lambda s: 42)
    calls = []
    monkeypatch.setattr(im, "_run_gh", lambda a: calls.append(a) or "")
    im.execute_action(im.ACTION_CLOSE, target_session="2026-09-29",
                       reason="x", run_id="1")
    assert any("close" in c and "42" in c for c in calls)


def test_im_execute_action_ensure_failure_comenta(monkeypatch, capsys):
    monkeypatch.setattr(im, "_ensure_label", lambda: None)
    monkeypatch.setattr(im, "_find_issue_by_title", lambda s: 42)
    calls = []
    monkeypatch.setattr(im, "_run_gh", lambda a: calls.append(a) or "")
    im.execute_action(im.ACTION_ENSURE_FAILURE,
                       target_session="2026-09-29",
                       reason="red", run_id="1")
    assert any("comment" in c for c in calls)


def test_im_execute_action_ensure_failure_crea(monkeypatch, capsys):
    monkeypatch.setattr(im, "_ensure_label", lambda: None)
    monkeypatch.setattr(im, "_find_issue_by_title", lambda s: None)
    fake = MagicMock(returncode=0, stdout="https://.../issues/1")
    monkeypatch.setattr(im.subprocess, "run", lambda *a, **k: fake)
    im.execute_action(im.ACTION_ENSURE_FAILURE,
                       target_session="2026-09-29",
                       reason="red", run_id="1")
    out = capsys.readouterr().out
    assert "creado" in out or "issue creado" in out
