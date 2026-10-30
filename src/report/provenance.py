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

import hashlib
import os
import subprocess
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


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
    # Fix 2026-10-30: el literal "10/10" era hardcoded y mentia cuando
    # el gate tenia 11 checks (2026-10-17). Ahora se publica el conteo
    # real. Contrato: N/N con N = numero actual de checks del gate.
    if passed and errors_n == 0:
        return "{0}/{0}".format(checks_n)
    return "{0}/{1}".format(checks_n - errors_n, checks_n)


def _temporal_coherence(target_session, effective_date_equity,
                        contract_max_lag_days=0):
    """H8 (2026-10-16, auditor): valida coherencia temporal del bloque.

    Si target_session == effective_date -> 'OK'.
    Si effective_date < target_session por <= max_lag_days -> 'LAG_ADMITTED'.
    Si effective_date < target_session por > max_lag_days -> 'INCOHERENT'.
    Si alguno falta -> 'N/D'.
    """
    if not target_session or not effective_date_equity:
        return "N/D"
    try:
        t = pd.Timestamp(target_session).date()
        e = pd.Timestamp(effective_date_equity).date()
    except Exception:
        return "N/D"
    if t == e:
        return "OK"
    if e > t:
        return "ANOMALY"
    lag = (t - e).days
    if lag <= contract_max_lag_days:
        return "LAG_ADMITTED"
    return "INCOHERENT"


def _make_local_run_id(reference_date, commit, generated_at):
    """H2 (2026-10-17): ID determinista 'local-<8 hex>' para
    ejecuciones locales (sin GITHUB_RUN_ID valido)."""
    if hasattr(reference_date, "isoformat"):
        ref_str = reference_date.isoformat()
    else:
        ref_str = str(reference_date or "NONE")
    seed = "{0}|{1}|{2}".format(ref_str, commit or "NO_COMMIT", generated_at)
    return "local-{0}".format(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8])


def _is_gate_passed(gate_str):
    """True si el string es 'N/N' (passed). Formato emitido por
    _compute_validation_gate. Acepta cualquier N >= 1."""
    if not isinstance(gate_str, str):
        return False
    parts = gate_str.split("/")
    if len(parts) != 2:
        return False
    return parts[0] == parts[1]


def build_provenance(
    reference_date,
    *,
    target_session=None,
    effective_date_equity=None,
    validation_gate_result=None,
    contract_max_lag_days=0,
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
    # H2 (2026-10-17): si no hay GITHUB_RUN_ID valido, generar
    # identificador local determinista.
    if not run_id or str(run_id) == "0":
        run_id = _make_local_run_id(reference_date, commit, generated_at)
    _gate = _compute_validation_gate(validation_gate_result)
    _coh = _temporal_coherence(
        target_session, effective_date_equity,
        contract_max_lag_days=contract_max_lag_days,
    )
    # H8 (2026-10-16, auditor): coherencia temporal. Si INCOHERENT con
    # validation_gate=10/10, es contradiccion: el gate no deberia haber
    # pasado. Se marca explicitamente para que el consumidor lo vea.
    if _coh == "INCOHERENT" and _gate and _is_gate_passed(_gate):
        _gate = _gate + " (INCOHERENT)"
    return {
        "commit_sha": commit,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "target_session": target_session,
        "reference_date": reference_date,
        "effective_date_equity": effective_date_equity,
        "contract_max_lag_days": contract_max_lag_days,
        "temporal_coherence": _coh,
        "validation_gate": _gate,
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
    # H2 (2026-10-17): tres ramas: local-<hash>, CI run_id, N/D.
    if run_id and str(run_id).startswith("local-"):
        out.append("- `run_id`: {0} (ejecucion local)\n".format(run_id))
    elif run_id and str(run_id) != "0":
        attempt_str = (
            " (attempt {0})".format(run_attempt) if run_attempt else ""
        )
        out.append("- `run_id`: {0}{1}\n".format(run_id, attempt_str))
    else:
        out.append("- `run_id`: N/D (no identificado)\n")

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
    _coh = prov.get("temporal_coherence")
    if _coh and _coh != "N/D":
        out.append("- `temporal_coherence`: {0}\n".format(_coh))
        # P6 (2026-10-22, auditor externo): aclarar alcance.
        out.append("*`temporal_coherence` valida solo `stock_prices` y "
                   "`market_data` (sesion esperada vs ultimo dato). El "
                   "resto de fuentes (SSGA, BME, Euronext, FINRA, SEC) "
                   "declara su lag y frescura en Data Freshness. "
                   "Coherencia temporal de los dos parquets no implica "
                   "sincronizacion de todas las fuentes.*\n")
    out.append("- `generated_at`: {0} (UTC)\n".format(
        _fmt(prov.get("generated_at"))))

    out.append("\n")
    out.append(
        "*Este bloque identifica la ejecucion que genero el reporte. "
        "Permite certificar que la salida corresponde a un commit "
        "concreto y a una sesion objetivo concreta.*\n\n"
    )
    # REP-AUDIT-20261020-REPORT-REVIEW-EXTERNAL (P2 auditor externo
    # 2026-10-20): reference_date es la referencia de ejecucion del
    # pipeline, no el corte estricto de datos. Fuentes externas (CBOE,
    # FRED, Yahoo) pueden incorporarse despues de esa marca.
    out.append(
        "*`reference_date` es la referencia de ejecucion del pipeline, "
        "no un corte estricto de datos. Las fuentes externas pueden "
        "incorporarse despues de esa marca. La fecha "
        "efectiva de cada mercado se declara en `effective_date_equity` "
        "y en Data Freshness.*\n\n"
    )
    return out