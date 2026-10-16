# -*- coding: utf-8 -*-
"""Provenance del reporte diario (H8, dictamen 2026-10-16).

Bloque de trazabilidad que identifica la ejecucion que genero el
reporte: commit, run de CI, sesion objetivo, fechas efectivas. Sin
este bloque, un reporte no puede certificarse (no se puede distinguir
un artefacto reciente de una regresion ni de una salida generada
antes de las correcciones).

Contrato:
  - Bloque informativo, no decisorio.
  - Campos opcionales. Ausentes -> 'N/D'.
  - El commit viene de GITHUB_SHA (primeros 8) o git rev-parse HEAD.
  - target_session es responsabilidad del caller (run.py lo calcula
    con last_expected_market_date sobre reference_date).
  - validation_gate se deriva del resultado del run_validation_gate.
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from typing import Optional


def _git_head_short() -> Optional[str]:
    """Devuelve primeros 8 chars del HEAD via git rev-parse."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except Exception:
        pass
    return None


def _fmt(v, default="N/D"):
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _compute_validation_gate(vg):
    """'10/10' si passed y sin errores; 'N/M' si no. None si vg ausente."""
    if not isinstance(vg, dict):
        return None
    passed = bool(vg.get("passed"))
    errors_n = len(vg.get("errors") or [])
    checks_n = len(vg.get("checks") or [])
    if passed and errors_n == 0:
        return "10/10" if checks_n >= 10 else "{0}/{0}".format(checks_n)
    return "{0}/{1}".format(checks_n - errors_n, checks_n)


def build_provenance(
    reference_date,
    *,
    target_session=None,
    effective_date_equity=None,
    validation_gate_result=None,
    env=None,
    generated_at=None,
) -> dict:
    """Construye el dict de provenance desde fuentes disponibles.

    Args:
        reference_date: datetime del pipeline.
        target_session: str ISO o None.
        effective_date_equity: date | str | None.
        validation_gate_result: dict con passed, checks, errors, o None.
        env: dict de env vars. Default os.environ.
        generated_at: str ISO. Default now UTC.

    Returns:
        dict listo para render_provenance.
    """
    e = env if env is not None else os.environ
    commit = (e.get("GITHUB_SHA") or "")[:8] or None
    run_id = e.get("GITHUB_RUN_ID") or None
    run_attempt = e.get("GITHUB_RUN_ATTEMPT") or None
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    return {
        "commit_sha": commit,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "target_session": target_session,
        "reference_date": reference_date,
        "effective_date_equity": effective_date_equity,
        "validation_gate": _compute_validation_gate(validation_gate_result),
        "generated_at": generated_at,
    }


def render_provenance(prov: Optional[dict]) -> list:
    """Renderiza el bloque Provenance.

    Devuelve lista de lineas markdown. Sin side effects.
    Si prov es None/vacio, devuelve lista vacia (compat).
    """
    if not prov:
        return []
    out = []
    out.append("**Provenance:**\n")

    commit = prov.get("commit_sha") or _git_head_short()
    out.append("- `commit_sha`: {0}\n".format(_fmt(commit)))

    run_id = prov.get("run_id")
    run_attempt = prov.get("run_attempt")
    if run_id and str(run_id) != "0":
        attempt_str = (
            " (attempt {0})".format(run_attempt) if run_attempt else ""
        )
        out.append("- `run_id`: {0}{1}\n".format(run_id, attempt_str))
    else:
        out.append("- `run_id`: N/D (ejecucion local)\n")

    out.append("- `target_session`: {0}\n".format(
        _fmt(prov.get("target_session"))))

    ref = prov.get("reference_date")
    if isinstance(ref, datetime):
        out.append("- `reference_date`: {0}\n".format(ref.isoformat()))
    else:
        out.append("- `reference_date`: {0}\n".format(_fmt(ref)))

    out.append("- `effective_date_equity`: {0}\n".format(
        _fmt(prov.get("effective_date_equity"))))
    out.append("- `validation_gate`: {0}\n".format(
        _fmt(prov.get("validation_gate"))))
    out.append("- `generated_at`: {0} (UTC)\n".format(
        _fmt(prov.get("generated_at"))))

    out.append("\n")
    out.append(
        "*Este bloque identifica la ejecucion que genero el reporte. "
        "Permite certificar que la salida corresponde a un commit "
        "concreto y a una sesion objetivo concreta.*\n\n"
    )
    return out