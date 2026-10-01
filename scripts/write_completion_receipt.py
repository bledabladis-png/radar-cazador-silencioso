#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Escribe el Completion Receipt (contrato v1, 2026-10-01).

Ejecutado por daily_run.yml tras guard_coverage. Lee:
- data/stock_prices.parquet.manifest.json
- outputs/state/validation_gate_result.json (escrito por run.py)
- env: GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, GITHUB_SHA
- git rev-parse HEAD (fallback si no hay env)

Escribe outputs/state/completion_receipt.json con el schema del
contrato (docs/auditoria/daily_run_gate_contrato_v1.md §3).

Exit 0 si escribe correctamente. Exit 1 si falta informacion critica
(el yml falla con if: success() y no sube artifact, lo que es
fail-safe: el siguiente slot hara READY).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = ROOT / "data" / "stock_prices.parquet.manifest.json"
VG_RESULT_PATH = ROOT / "outputs" / "state" / "validation_gate_result.json"
OUT_PATH = ROOT / "outputs" / "state" / "completion_receipt.json"

SCHEMA_VERSION = 1
WORKFLOW_NAME = "daily_run"


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


def main():
    manifest = _read_json(MANIFEST_PATH)
    if not isinstance(manifest, dict):
        print("[ERROR] manifest no legible: {0}".format(MANIFEST_PATH))
        return 1

    vg = _read_json(VG_RESULT_PATH)
    if not isinstance(vg, dict):
        print("[ERROR] validation_gate_result no legible: {0}".format(
            VG_RESULT_PATH
        ))
        return 1

    passed = bool(vg.get("passed"))
    errors_n = int(vg.get("errors_n") or 0)
    checks_n = int(vg.get("checks_n") or 0)

    # validation_gate: "10/10" solo si passed y errors_n == 0. El
    # gate rechaza cualquier otro valor (contrato v1 §3).
    validation_gate = "10/10" if (passed and errors_n == 0) else "{0}/{1}".format(
        checks_n - errors_n, checks_n
    )

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
        "guard_coverage": "OK",
        "validation_gate": validation_gate,
        "pipeline_conclusion": "success",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("[OK] receipt escrito: {0}".format(OUT_PATH))
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
