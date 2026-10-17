#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Registro diagnostico de intentos del pipeline.

Enmienda v1.1 (2026-10-16): cada ejecucion del pipeline registra
una linea en outputs/state/completion_attempt.jsonl con el outcome
(SUCCESS / INCOMPLETE / BLOCKED) y las causas. Se sube como artifact
por ejecucion (no se commitea al repo).

Uso: invocado desde daily_run.yml con `if: always()`.
Lee env: GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, GITHUB_SHA,
         GITHUB_EVENT_NAME, TARGET_SESSION (del gate).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.temporal_contracts.validation import (
    validate_artifact_contracts,
    STATUS_BLOCKED as V_BLOCKED,
    STATUS_UNAVAILABLE as V_UNAVAILABLE,
)

MANIFESTS = (
    "data/stock_prices.parquet.manifest.json",
    "data/market_data.parquet.manifest.json",
)

OUT_PATH = ROOT / "outputs" / "state" / "completion_attempt.jsonl"


def _read_json(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sha256_file(path):
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_failed_contracts():
    """Devuelve (failed_contracts, reasons, manifest_hashes)."""
    failed = []
    reasons = []
    hashes = {}
    for mpath in MANIFESTS:
        m = _read_json(mpath)
        if m is None:
            failed.append(mpath + ":manifest_missing")
            reasons.append("manifest no legible: " + mpath)
            hashes[mpath] = ""
            continue
        hashes[mpath] = _sha256_file(mpath)
        v = validate_artifact_contracts(m)
        if v.status in (V_BLOCKED, V_UNAVAILABLE):
            failed.append(mpath + ":BLOCKED")
            for r in v.reasons:
                reasons.append(mpath + ": " + r)
    return failed, reasons, hashes


def _write_entry(outcome, failed, reasons, hashes, target_session):
    run_id = os.environ.get("GITHUB_RUN_ID", "0")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    commit_sha = os.environ.get("GITHUB_SHA", "")
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    entry = {
        "run_id": int(run_id) if run_id.isdigit() else 0,
        "run_attempt": int(run_attempt) if run_attempt.isdigit() else 1,
        "target_session": target_session or "",
        "attempt_timestamp": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "outcome": outcome,
        "failed_contracts": failed,
        "reasons": reasons,
        "manifest_hashes": hashes,
        "commit_sha": commit_sha,
        "event": event,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry



def _read_guard_result():
    """Lee y valida guard_coverage_result.json (H20, dictamen 2026-10-17).

    Devuelve dict con state: 'ok' | 'failed' | 'unavailable' | 'absent'.
    Solo 'failed' y 'unavailable' son estados de bloqueo.
    """
    path = ROOT / "outputs" / "state" / "guard_coverage_result.json"
    if not path.exists():
        return {"state": "absent", "reason": "", "failures": []}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {
            "state": "unavailable",
            "reason": "guard_coverage_result.json ilegible: {0}: {1}".format(
                type(e).__name__, e
            ),
            "failures": [],
        }
    if not isinstance(d, dict):
        return {
            "state": "unavailable",
            "reason": "guard_coverage_result.json no es dict",
            "failures": [],
        }
    if d.get("schema_version") != 1:
        return {
            "state": "unavailable",
            "reason": "guard_coverage_result.json schema_version={0!r} != 1".format(
                d.get("schema_version")
            ),
            "failures": [],
        }
    if d.get("status") == "running":
        return {
            "state": "unavailable",
            "reason": "guard_coverage_result.json status=running (guard interrumpido a mitad)",
            "failures": [],
        }
    run_id_env = os.environ.get("GITHUB_RUN_ID", "0")
    run_attempt_env = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    expected_run_id = int(run_id_env) if run_id_env.isdigit() else 0
    expected_run_attempt = int(run_attempt_env) if run_attempt_env.isdigit() else 1
    if d.get("run_id") != expected_run_id:
        return {
            "state": "unavailable",
            "reason": "guard_coverage_result.json run_id={0} != actual={1}".format(
                d.get("run_id"), expected_run_id
            ),
            "failures": [],
        }
    if d.get("run_attempt") != expected_run_attempt:
        return {
            "state": "unavailable",
            "reason": "guard_coverage_result.json run_attempt={0} != actual={1}".format(
                d.get("run_attempt"), expected_run_attempt
            ),
            "failures": [],
        }
    ok = d.get("ok")
    if not isinstance(ok, bool):
        return {
            "state": "unavailable",
            "reason": "guard_coverage_result.json sin campo ok booleano",
            "failures": [],
        }
    failures = d.get("failures") or []
    if not isinstance(failures, list):
        failures = [str(failures)]
    if ok:
        return {"state": "ok", "reason": "", "failures": [str(f) for f in failures]}
    return {"state": "failed", "reason": "", "failures": [str(f) for f in failures]}


def _detect_outcome():
    """Determina el outcome segun el estado de los manifests y del receipt."""
    receipt_path = ROOT / "outputs" / "state" / "completion_receipt.json"
    if receipt_path.exists():
        return "SUCCESS", [], [], {}

    # Sin receipt: determinar razon
    failed, reasons, hashes = _collect_failed_contracts()

    # H20 + escenario 5 del dictamen externo (2026-10-17): leer el
    # guard ANTES del branch VG, para que la causa del guard quede
    # disponible aunque VG decida la etiqueta final. Un JSON corrupto,
    # un schema distinto, un status=running (guard interrumpido a
    # mitad) o una identidad de run distinta no cuentan como evidencia
    # valida: se registra GUARD_COVERAGE_UNAVAILABLE, no se ignora.
    guard = _read_guard_result()

    # Si hay validation_gate_result con passed=False, es INCOMPLETE por gate.
    vg = _read_json(ROOT / "outputs" / "state" / "validation_gate_result.json")
    if isinstance(vg, dict) and vg.get("passed") is False:
        reasons.insert(0, "validation_gate passed=False")
        # Precedencia INCOMPLETE, pero ambas causas disponibles.
        if guard["state"] == "failed":
            reasons.append("guard_coverage: {0} fallo(s)".format(len(guard["failures"])))
            for f in guard["failures"]:
                reasons.append(str(f))
        elif guard["state"] == "unavailable":
            reasons.append("guard_coverage: resultado no valido")
            reasons.append(guard["reason"])
        return "INCOMPLETE", failed, reasons, hashes

    if guard["state"] == "failed":
        failed.append("GUARD_COVERAGE")
        reasons.insert(0, "guard_coverage: {0} fallo(s)".format(len(guard["failures"])))
        for f in guard["failures"]:
            reasons.append(str(f))
        return "BLOCKED", failed, reasons, hashes
    if guard["state"] == "unavailable":
        failed.append("GUARD_COVERAGE_UNAVAILABLE")
        reasons.insert(0, "guard_coverage: resultado no valido")
        reasons.append(guard["reason"])
        return "BLOCKED", failed, reasons, hashes

    if failed:
        return "BLOCKED", failed, reasons, hashes

    return "INCOMPLETE", [], reasons or ["sin receipt y sin causa identificada"], hashes


def main():
    target_session = os.environ.get("TARGET_SESSION", "")
    outcome, failed, reasons, hashes = _detect_outcome()
    entry = _write_entry(outcome, failed, reasons, hashes, target_session)
    print("[OK] completion_attempt.jsonl:")
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())