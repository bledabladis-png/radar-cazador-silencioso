#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Escribe el Completion Receipt (contrato v1 + enmienda v1.1).

Precondiciones de emision (fail-closed):
  1. manifest legible.
  2. validation_gate_result.json con passed=True, errors_n=0, checks_n>=10.
  3. written_at reciente (<=6h).
  4. HD-4: validation strict (ya validado por _validate_vg_result).
  5. Enmienda v1.1: validate_artifact_contracts() no devuelve BLOCKED
     ni UNAVAILABLE. Si devuelve alguno, no se emite receipt.

Schema del receipt: v1 sin cambios.
"""
from __future__ import annotations

import json
import os
import subprocess
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

MANIFEST_PATH = ROOT / "data" / "stock_prices.parquet.manifest.json"
VG_RESULT_PATH = ROOT / "outputs" / "state" / "validation_gate_result.json"
OUT_PATH = ROOT / "outputs" / "state" / "completion_receipt.json"

SCHEMA_VERSION = 1
WORKFLOW_NAME = "daily_run"

RECEIPT_MAX_AGE_HOURS = 6
VG_MIN_CHECKS = 10


def _read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _git_head():
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(ROOT),
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def _parse_written_at(s):
    if not isinstance(s, str) or not s:
        return None
    try:
        iso = s[:-1] + "+00:00" if s.endswith("Z") else s
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _validate_vg_result(vg):
    if not isinstance(vg, dict):
        return False, "VG result no es dict"
    if vg.get("passed") is not True:
        return False, "passed != True (estricto)"
    errors_n = vg.get("errors_n")
    if not isinstance(errors_n, int) or isinstance(errors_n, bool) or errors_n != 0:
        return False, "errors_n != 0"
    checks_n = vg.get("checks_n")
    if not isinstance(checks_n, int) or isinstance(checks_n, bool) or checks_n < VG_MIN_CHECKS:
        return False, "checks_n < {0}".format(VG_MIN_CHECKS)
    written_at = _parse_written_at(vg.get("written_at"))
    if written_at is None:
        return False, "written_at ausente o invalido"
    now_utc = datetime.now(timezone.utc)
    age_h = (now_utc - written_at).total_seconds() / 3600
    if age_h < -0.1:
        return False, "written_at en el futuro"
    if age_h > RECEIPT_MAX_AGE_HOURS:
        return False, "written_at demasiado antiguo ({0:.1f}h)".format(age_h)
    return True, "OK"


def main():
    manifest = _read_json(MANIFEST_PATH)
    if not isinstance(manifest, dict):
        print("[ERROR] manifest no legible: {0}".format(MANIFEST_PATH))
        return 1

    vg = _read_json(VG_RESULT_PATH)
    if not isinstance(vg, dict):
        print("[ERROR] validation_gate_result no legible: {0}".format(VG_RESULT_PATH))
        return 1

    vg_ok, vg_reason = _validate_vg_result(vg)
    if not vg_ok:
        print("[ERROR] VG result no valido: {0}".format(vg_reason))
        return 1

    # Enmienda v1.1: validacion contractual compartida (misma funcion
    # que usa guard_coverage.py).
    cv = validate_artifact_contracts(manifest)
    if cv.status in (V_BLOCKED, V_UNAVAILABLE):
        print("[ERROR] Validacion contractual: {0}".format(cv.status))
        for r in cv.reasons:
            print("  - {0}".format(r))
        print("[ERROR] No se emite receipt (fail-closed).")
        return 1
    print("[OK] Validacion contractual: {0}".format(cv.status))

    checks_n = int(vg["checks_n"])
    errors_n = int(vg["errors_n"])
    guard_coverage = "OK"
    validation_gate = "10/10"

    quality = manifest.get("quality") or {}
    artifact = manifest.get("artifact") or {}

    target_session = quality.get("expected_session")
    if not isinstance(target_session, str) or not target_session:
        print("[ERROR] manifest sin expected_session")
        return 1

    manifest_sha = artifact.get("sha256") or ""
    coverage = quality.get("coverage_pct_last")
    if not isinstance(coverage, (int, float)):
        coverage = 0.0

    run_id = os.environ.get("GITHUB_RUN_ID", "0")
    try:
        run_id_int = int(run_id)
    except Exception:
        run_id_int = 0
    if run_id_int == 0:
        print("[ERROR] GITHUB_RUN_ID invalido o ausente: {0!r}".format(run_id))
        return 1

    run_attempt_raw = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    try:
        run_attempt = int(run_attempt_raw)
    except Exception:
        run_attempt = 1

    commit_sha = os.environ.get("GITHUB_SHA") or _git_head() or ""

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "target_session": target_session,
        "workflow": WORKFLOW_NAME,
        "run_id": run_id_int,
        "run_attempt": run_attempt,
        "completed_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "commit_sha": commit_sha,
        "manifest_sha256": manifest_sha,
        "manifest_coverage_pct": float(coverage),
        "guard_coverage": guard_coverage,
        "validation_gate": validation_gate,
        "pipeline_conclusion": "success",
        "success_path_reached": True,
        "validation_gate_checks_n": checks_n,
        "validation_gate_errors_n": errors_n,
        "contract_validation_status": cv.status,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("[OK] receipt escrito: {0}".format(OUT_PATH))
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())