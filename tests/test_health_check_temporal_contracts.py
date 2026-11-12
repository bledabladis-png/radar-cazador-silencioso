# -*- coding: utf-8 -*-
"""Tests de check_temporal_contracts_result (S35-bis)."""
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "health_check",
    str(ROOT / "scripts" / "health_check.py"),
)
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)


def _path(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    p = tmp_path / "outputs" / "state"
    p.mkdir(parents=True)
    return p / "temporal_contracts_result.json"


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _payload(**overrides):
    bc = {f"C{i}": {"status": "OK"} for i in range(15)}
    base = {
        "schema_version": 1,
        "contracts_total": 15,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "by_contract": bc,
    }
    base.update(overrides)
    return base


def _patch_registry(monkeypatch, n=15):
    monkeypatch.setattr(
        "src.temporal_contracts.list_contracts",
        lambda: list(range(n)),
    )


def test_json_ausente_warn(tmp_path, monkeypatch):
    _path(tmp_path, monkeypatch)
    _patch_registry(monkeypatch)
    r = hc.check_temporal_contracts_result()
    assert len(r) == 1
    assert r[0].status == hc.WARN
    assert "ausente" in r[0].message


def test_json_invalido_fail(tmp_path, monkeypatch):
    p = _path(tmp_path, monkeypatch)
    p.write_text("not json", encoding="utf-8")
    r = hc.check_temporal_contracts_result()
    assert len(r) == 1
    assert r[0].status == hc.FAIL
    assert "invalido" in r[0].message


def test_json_no_dict_fail(tmp_path, monkeypatch):
    p = _path(tmp_path, monkeypatch)
    p.write_text("[1,2,3]", encoding="utf-8")
    r = hc.check_temporal_contracts_result()
    assert len(r) == 1
    assert r[0].status == hc.FAIL


def test_schema_version_roto_fail(tmp_path, monkeypatch):
    p = _path(tmp_path, monkeypatch)
    _patch_registry(monkeypatch)
    _write(p, _payload(schema_version=2))
    r = hc.check_temporal_contracts_result()
    assert len(r) == 1
    assert r[0].status == hc.FAIL
    assert "schema_version" in r[0].message


def test_by_contract_vacio_fail(tmp_path, monkeypatch):
    p = _path(tmp_path, monkeypatch)
    _patch_registry(monkeypatch)
    _write(p, _payload(by_contract={}))
    r = hc.check_temporal_contracts_result()
    assert len(r) == 1
    assert r[0].status == hc.FAIL
    assert "by_contract" in r[0].message


def test_drift_contracts_total_warn(tmp_path, monkeypatch):
    p = _path(tmp_path, monkeypatch)
    _patch_registry(monkeypatch, n=20)
    _write(p, _payload(contracts_total=15))
    r = hc.check_temporal_contracts_result()
    assert any(x.status == hc.WARN and "drift" in x.message for x in r)


def test_stale_json_warn(tmp_path, monkeypatch):
    p = _path(tmp_path, monkeypatch)
    _patch_registry(monkeypatch)
    old = datetime.now(timezone.utc) - timedelta(hours=100)
    _write(p, _payload(generated_at_utc=old.isoformat()))
    r = hc.check_temporal_contracts_result()
    assert any(x.status == hc.WARN and "sin refrescar" in x.message for x in r)


def test_json_fresco_ok(tmp_path, monkeypatch):
    p = _path(tmp_path, monkeypatch)
    _patch_registry(monkeypatch)
    _write(p, _payload())
    r = hc.check_temporal_contracts_result()
    assert len(r) == 1
    assert r[0].status == hc.OK
    assert "15 contratos" in r[0].message


def test_blocked_fail(tmp_path, monkeypatch):
    p = _path(tmp_path, monkeypatch)
    _patch_registry(monkeypatch)
    bc = {f"C{i}": {"status": "OK"} for i in range(14)}
    bc["C14"] = {"status": "BLOCKED"}
    _write(p, _payload(by_contract=bc))
    r = hc.check_temporal_contracts_result()
    assert any(x.status == hc.FAIL and "BLOCKED" in x.message for x in r)


def test_pending_warn(tmp_path, monkeypatch):
    p = _path(tmp_path, monkeypatch)
    _patch_registry(monkeypatch)
    bc = {f"C{i}": {"status": "OK"} for i in range(14)}
    bc["C14"] = {"status": "PENDING"}
    _write(p, _payload(by_contract=bc))
    r = hc.check_temporal_contracts_result()
    assert any(x.status == hc.WARN and "PENDING" in x.message for x in r)


def test_insufficient_warn(tmp_path, monkeypatch):
    p = _path(tmp_path, monkeypatch)
    _patch_registry(monkeypatch)
    bc = {f"C{i}": {"status": "OK"} for i in range(14)}
    bc["C14"] = {"status": "INSUFFICIENT"}
    _write(p, _payload(by_contract=bc))
    r = hc.check_temporal_contracts_result()
    assert any(x.status == hc.WARN and "INSUFFICIENT" in x.message for x in r)


def test_unknown_status_warn(tmp_path, monkeypatch):
    p = _path(tmp_path, monkeypatch)
    _patch_registry(monkeypatch)
    bc = {f"C{i}": {"status": "OK"} for i in range(14)}
    bc["C14"] = {"status": "WEIRD"}
    _write(p, _payload(by_contract=bc))
    r = hc.check_temporal_contracts_result()
    assert any(x.status == hc.WARN and "desconocido" in x.message for x in r)
