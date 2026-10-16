# -*- coding: utf-8 -*-
"""Tests del registro de intentos (enmienda v1.1, commit 4)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.write_completion_attempt as wca


def _setup_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("TARGET_SESSION", "2026-10-15")
    monkeypatch.setattr(wca, "OUT_PATH", tmp_path / "attempt.jsonl")


def _mk_manifest(tmp_path, name, by_market, status="VALID_WITH_MISSING",
                 last="2026-10-15", expected="2026-10-15", cov=0.99):
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.parquet.manifest.json"
    p.write_text(json.dumps({
        "artifact": {"sha256": "abc"*20},
        "quality": {
            "status": status,
            "last_date": last,
            "expected_session": expected,
            "coverage_pct_last": cov,
            "by_market": by_market,
        },
    }), encoding="utf-8")
    return p


def test_success_si_existe_receipt(tmp_path, monkeypatch):
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    # Coloca un receipt que simula exito
    (tmp_path / "outputs" / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs" / "state" / "completion_receipt.json").write_text("{}")
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    monkeypatch.setattr(wca, "MANIFESTS", ())
    rc = wca.main()
    assert rc == 0
    lines = (tmp_path / "attempt.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["outcome"] == "SUCCESS"
    assert entry["target_session"] == "2026-10-15"


def test_blocked_por_contrato_invalido(tmp_path, monkeypatch):
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    _mk_manifest(tmp_path, "stock_prices",
                 {"US_EQUITY": {"n": 255, "status": "INVALID",
                                 "coverage_at_session": 0.0}})
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    monkeypatch.setattr(wca, "MANIFESTS", (
        "data/stock_prices.parquet.manifest.json",
    ))
    rc = wca.main()
    assert rc == 0
    lines = (tmp_path / "attempt.jsonl").read_text(encoding="utf-8").strip().splitlines()
    entry = json.loads(lines[0])
    assert entry["outcome"] == "BLOCKED"
    assert any("BLOCKED" in f for f in entry["failed_contracts"])
    assert entry["manifest_hashes"]


def test_incomplete_si_vg_falla(tmp_path, monkeypatch):
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs" / "state" / "validation_gate_result.json").write_text(
        json.dumps({"passed": False, "checks_n": 10, "errors_n": 2})
    )
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    monkeypatch.setattr(wca, "MANIFESTS", ())
    wca.main()
    entry = json.loads(
        (tmp_path / "attempt.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert entry["outcome"] == "INCOMPLETE"


def test_append_no_sobrescribe(tmp_path, monkeypatch):
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    monkeypatch.setattr(wca, "MANIFESTS", ())
    wca.main()
    wca.main()
    lines = (tmp_path / "attempt.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_campos_obligatorios(tmp_path, monkeypatch):
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    monkeypatch.setattr(wca, "MANIFESTS", ())
    wca.main()
    entry = json.loads(
        (tmp_path / "attempt.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    for k in ("run_id", "run_attempt", "target_session",
              "attempt_timestamp", "outcome", "failed_contracts",
              "reasons", "manifest_hashes"):
        assert k in entry, "falta campo {0}".format(k)
    assert entry["run_id"] == 12345
    assert entry["run_attempt"] == 1