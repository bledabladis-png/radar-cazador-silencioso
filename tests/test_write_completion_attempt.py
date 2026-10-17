# -*- coding: utf-8 -*-
"""Tests del registro de intentos (enmienda v1.1, commit 4)."""
import json
import os
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


def test_outcome_incomplete_prioritario_sobre_blocked(tmp_path, monkeypatch):
    """VG failed + contrato bloqueado -> INCOMPLETE, pero la causa BLOCKED se conserva."""
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    _mk_manifest(tmp_path, "stock_prices",
                 {"US_EQUITY": {"n": 255, "status": "INVALID",
                                "coverage_at_session": 0.0}})
    monkeypatch.setattr(wca, "MANIFESTS", (
        "data/stock_prices.parquet.manifest.json",
    ))
    (tmp_path / "outputs" / "state").mkdir(parents=True)
    (tmp_path / "outputs" / "state" / "validation_gate_result.json").write_text(
        json.dumps({"passed": False, "checks_n": 10, "errors_n": 2}),
        encoding="utf-8",
    )

    rc = wca.main()
    assert rc == 0
    entry = json.loads(
        (tmp_path / "attempt.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert entry["outcome"] == "INCOMPLETE", (
        "VG failed debe ganar la etiqueta (pipeline no completo)"
    )
    assert any("validation_gate passed=False" in r for r in entry["reasons"])
    assert entry["failed_contracts"], (
        "el bloqueo del contrato debe seguir visible en failed_contracts"
    )
    assert any("BLOCKED" in f for f in entry["failed_contracts"])


def test_outcome_blocked_por_contrato_unavailable(tmp_path, monkeypatch):
    """Contrato UNAVAILABLE (by_market sin el critico) -> BLOCKED fail-closed."""
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    # by_market con un mercado NO critico; US_EQUITY ausente -> UNAVAILABLE.
    _mk_manifest(tmp_path, "stock_prices",
                 {"BME": {"n": 50, "status": "VALID", "coverage_at_session": 0.99}})
    monkeypatch.setattr(wca, "MANIFESTS", (
        "data/stock_prices.parquet.manifest.json",
    ))

    rc = wca.main()
    assert rc == 0
    entry = json.loads(
        (tmp_path / "attempt.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert entry["outcome"] == "BLOCKED"
    assert entry["failed_contracts"]
    assert any("US_EQUITY" in r for r in entry["reasons"]), (
        "la razon debe mencionar el contrato critico ausente"
    )


def test_outcome_blocked_por_manifest_missing(tmp_path, monkeypatch):
    """Manifest listado en MANIFESTS pero ausente en disco -> BLOCKED."""
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    monkeypatch.setattr(wca, "MANIFESTS", (
        "data/no_existe.parquet.manifest.json",
    ))

    rc = wca.main()
    assert rc == 0
    entry = json.loads(
        (tmp_path / "attempt.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert entry["outcome"] == "BLOCKED"
    assert any("manifest_missing" in f for f in entry["failed_contracts"])
    assert any("manifest no legible" in r for r in entry["reasons"])


def test_failed_contracts_no_duplica_con_multiples_reasons(tmp_path, monkeypatch):
    """Un manifest con N reasons BLOCKED -> 1 sola entrada en failed_contracts.

    Regresion 2026-10-17: el append de failed_contracts estaba dentro
    del bucle de reasons. Si un contrato generaba 2 razones, la ruta
    aparecia 2 veces en failed_contracts. Ahora el contrato del
    artifact es "una ruta por manifest fallido".
    """
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    # Manifest presente (no manifest_missing) para que entre en el
    # bloque de validate_artifact_contracts.
    _mk_manifest(tmp_path, "stock_prices",
                 {"US_EQUITY": {"n": 255, "status": "VALID",
                                "coverage_at_session": 0.99}})
    monkeypatch.setattr(wca, "MANIFESTS", (
        "data/stock_prices.parquet.manifest.json",
    ))

    # Mock: validate devuelve 2 reasons. Sin tocar el registry real.
    class _FakeV:
        status = "BLOCKED"
        reasons = ["razon 1", "razon 2"]

    import scripts.write_completion_attempt as _wca
    monkeypatch.setattr(_wca, "validate_artifact_contracts",
                        lambda m: _FakeV())

    rc = wca.main()
    assert rc == 0
    entry = json.loads(
        (tmp_path / "attempt.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert entry["outcome"] == "BLOCKED"
    assert len(entry["failed_contracts"]) == 1, (
        "failed_contracts debe tener 1 entrada por manifest, "
        "no 1 por razon. Encontrado: {0}".format(entry["failed_contracts"])
    )
    assert len(entry["reasons"]) == 2, (
        "reasons debe conservar las 2 razones (diagnostico completo)"
    )


# --- Addendum v1.2 (2026-10-17): attempt lee guard_coverage_result ---

def test_attempt_blocked_por_guard_result(tmp_path, monkeypatch):
    """Guard falla con identidad valida: attempt BLOCKED con el
    identificador estable GUARD_COVERAGE y detalles en reasons (H20)."""
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    monkeypatch.setattr(wca, "MANIFESTS", ())

    run_id_env = os.environ.get("GITHUB_RUN_ID", "0")
    run_attempt_env = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    expected_run_id = int(run_id_env) if run_id_env.isdigit() else 0
    expected_run_attempt = int(run_attempt_env) if run_attempt_env.isdigit() else 1

    (tmp_path / "outputs" / "state").mkdir(parents=True)
    (tmp_path / "outputs" / "state" / "guard_coverage_result.json").write_text(
        json.dumps({
            "schema_version": 1,
            "status": "failed",
            "run_id": expected_run_id,
            "run_attempt": expected_run_attempt,
            "target_session": "2026-10-15",
            "generated_at": "2026-10-17T06:00:00Z",
            "threshold": 0.95,
            "ok": False,
            "failures": [
                "data/stock_prices.parquet.manifest.json: "
                "coverage_pct_last=0.927052 < 0.95"
            ],
        }),
        encoding="utf-8",
    )

    rc = wca.main()
    assert rc == 0
    entry = json.loads(
        (tmp_path / "attempt.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert entry["outcome"] == "BLOCKED"
    assert "GUARD_COVERAGE" in entry["failed_contracts"]
    assert any("0.927" in r for r in entry["reasons"])


def test_attempt_vg_prioritario_sobre_guard(tmp_path, monkeypatch):
    """VG passed=False gana a guard ok=False (INCOMPLETE)."""
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    monkeypatch.setattr(wca, "MANIFESTS", ())

    run_id_env = os.environ.get("GITHUB_RUN_ID", "0")
    run_attempt_env = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    expected_run_id = int(run_id_env) if run_id_env.isdigit() else 0
    expected_run_attempt = int(run_attempt_env) if run_attempt_env.isdigit() else 1

    (tmp_path / "outputs" / "state").mkdir(parents=True)
    (tmp_path / "outputs" / "state" / "validation_gate_result.json").write_text(
        json.dumps({"passed": False, "checks_n": 10, "errors_n": 2}),
        encoding="utf-8",
    )
    (tmp_path / "outputs" / "state" / "guard_coverage_result.json").write_text(
        json.dumps({
            "schema_version": 1,
            "status": "failed",
            "run_id": expected_run_id,
            "run_attempt": expected_run_attempt,
            "target_session": "2026-10-15",
            "generated_at": "2026-10-17T06:00:00Z",
            "threshold": 0.95,
            "ok": False,
            "failures": ["x: y"],
        }),
        encoding="utf-8",
    )

    rc = wca.main()
    assert rc == 0
    entry = json.loads(
        (tmp_path / "attempt.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert entry["outcome"] == "INCOMPLETE"
    assert any("validation_gate" in r for r in entry["reasons"])


# --- H20 (dictamen externo 2026-10-17): validacion de identidad del guard ---

def _guard_payload(run_id=12345, run_attempt=1, schema=1,
                   status="failed", ok=False, failures=None):
    return {
        "schema_version": schema,
        "status": status,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "target_session": "2026-10-15",
        "generated_at": "2026-10-17T06:00:00Z",
        "threshold": 0.95,
        "ok": ok,
        "failures": failures if failures is not None else ["x: y"],
    }


def _setup_guard(tmp_path, monkeypatch, payload):
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    monkeypatch.setattr(wca, "MANIFESTS", ())
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    (tmp_path / "outputs" / "state").mkdir(parents=True)
    (tmp_path / "outputs" / "state" / "guard_coverage_result.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _entry(tmp_path):
    return json.loads(
        (tmp_path / "attempt.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )


def test_guard_run_id_mismatch_bloquea_con_unavailable(tmp_path, monkeypatch):
    """run_id distinto al actual -> GUARD_COVERAGE_UNAVAILABLE."""
    _setup_guard(tmp_path, monkeypatch, _guard_payload(run_id=99999))
    wca.main()
    entry = _entry(tmp_path)
    assert entry["outcome"] == "BLOCKED"
    assert "GUARD_COVERAGE_UNAVAILABLE" in entry["failed_contracts"]
    assert any("run_id" in r for r in entry["reasons"])


def test_guard_run_attempt_mismatch_bloquea(tmp_path, monkeypatch):
    """run_attempt distinto -> GUARD_COVERAGE_UNAVAILABLE."""
    _setup_guard(tmp_path, monkeypatch, _guard_payload(run_attempt=5))
    wca.main()
    entry = _entry(tmp_path)
    assert entry["outcome"] == "BLOCKED"
    assert "GUARD_COVERAGE_UNAVAILABLE" in entry["failed_contracts"]


def test_guard_status_running_bloquea(tmp_path, monkeypatch):
    """status=running (guard interrumpido) -> GUARD_COVERAGE_UNAVAILABLE."""
    _setup_guard(tmp_path, monkeypatch,
                 _guard_payload(status="running", ok=None))
    wca.main()
    entry = _entry(tmp_path)
    assert entry["outcome"] == "BLOCKED"
    assert "GUARD_COVERAGE_UNAVAILABLE" in entry["failed_contracts"]
    assert any("running" in r for r in entry["reasons"])


def test_guard_schema_invalido_bloquea(tmp_path, monkeypatch):
    """schema_version != 1 -> GUARD_COVERAGE_UNAVAILABLE."""
    _setup_guard(tmp_path, monkeypatch, _guard_payload(schema=2))
    wca.main()
    entry = _entry(tmp_path)
    assert entry["outcome"] == "BLOCKED"
    assert "GUARD_COVERAGE_UNAVAILABLE" in entry["failed_contracts"]
    assert any("schema_version" in r for r in entry["reasons"])


def test_guard_json_corrupto_bloquea(tmp_path, monkeypatch):
    """JSON invalido -> GUARD_COVERAGE_UNAVAILABLE."""
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    monkeypatch.setattr(wca, "MANIFESTS", ())
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    (tmp_path / "outputs" / "state").mkdir(parents=True)
    (tmp_path / "outputs" / "state" / "guard_coverage_result.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    wca.main()
    entry = _entry(tmp_path)
    assert entry["outcome"] == "BLOCKED"
    assert "GUARD_COVERAGE_UNAVAILABLE" in entry["failed_contracts"]


def test_guard_ok_no_bloquea(tmp_path, monkeypatch):
    """ok=True con identidad valida -> no aporta causa, sigue el flujo."""
    _setup_guard(tmp_path, monkeypatch,
                 _guard_payload(status="passed", ok=True, failures=[]))
    wca.main()
    entry = _entry(tmp_path)
    assert entry["outcome"] == "INCOMPLETE"
    assert entry["failed_contracts"] == []


def test_guard_ausente_no_bloquea(tmp_path, monkeypatch):
    """Fichero ausente -> no bloquea por guard; INCOMPLETE sin causa."""
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    monkeypatch.setattr(wca, "MANIFESTS", ())
    wca.main()
    entry = _entry(tmp_path)
    assert entry["outcome"] == "INCOMPLETE"
    assert entry["failed_contracts"] == []
