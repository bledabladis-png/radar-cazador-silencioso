#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate de disponibilidad para el pipeline diario.

Estados (dictamen auditor 2026-09-25, C3; contrato v1 2026-10-01):
  CURRENT    Existe completion receipt valido para target_session.
             No correr. Fuente unica: find_completion_receipt().
  READY      Sin receipt, probe OK. Correr pipeline.
  NOT_READY  Sin receipt, probe por debajo del threshold. Skip.
  ERROR      Fallo tecnico del probe. Skip.

Contrato v1 (docs/auditoria/daily_run_gate_contrato_v1.md):
- CURRENT depende EXCLUSIVAMENTE de un completion receipt inmutable
  publicado como artifact en GitHub Actions.
- _manifest_satisfies() se conserva como contrato independiente de
  integridad del artefacto. NO participa en la decision de CURRENT.

Concepto clave (C7): target_session es la sesion bursatil que el
pipeline intenta producir, NO el dia calendario UTC del slot.

Exit code: siempre 0. La decision viaja por outputs de GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.market_calendar import last_expected_market_date


# Panel operativo fijo. Muestra diversificada para detectar
# disponibilidad general del feed USA. NO es una estimacion
# estadisticamente representativa del universo completo (C4/C10).
GATE_PANEL_USA = (
    "AAPL", "MSFT", "JPM", "XOM", "JNJ",
    "WMT", "PG", "HD", "V", "UNH",
    "CVX", "ABBV", "KO", "PEP", "MRK",
    "COST", "BAC", "TMO", "AVGO", "MCD",
)

PROBE_MIN_COVERAGE = 0.90
MANIFEST_THRESHOLD = 0.95
MANIFEST_PATH = "data/stock_prices.parquet.manifest.json"

# Retry corto del probe para errores transitorios de red (auditor 2026-09-25).
# NO cubre "Yahoo aun no tiene el Close" (eso es NOT_READY, sin retry).
PROBE_MAX_RETRIES = 3
PROBE_RETRY_SLEEPS = (10, 30)

# Slots de cron declarados en .github/workflows/daily_run.yml (deben coincidir
# caracter por caracter). Fuente unica de verdad en Python.
CRON_SLOTS = (
    "17 23 * * *",
    "17 3 * * *",
    "17 5 * * *",
    "17 7 * * *",
    "17 11 * * *",
)
LAST_SLOT_CRON = "17 11 * * *"

# --- Completion Receipt (contrato v1, 2026-10-01) ---
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_ARTIFACT_PREFIX = "completion-receipt-"
# upload-artifact@v4 sube el basename del fichero cuando el path
# apunta a un fichero suelto. El yml sube
# outputs/state/completion_receipt.json -> dentro del ZIP el
# nombre es "completion_receipt.json".
RECEIPT_FILENAME = "completion_receipt.json"
RECEIPT_RETENTION_DAYS = 90
DEFAULT_REPO = "bledabladis-png/radar-cazador-silencioso"
GH_API_TIMEOUT = 15
GH_API_ZIP_TIMEOUT = 30
VALID_RECEIPT_STATUS = frozenset({"COMPLETED"})
VALID_RECEIPT_WORKFLOWS = frozenset({"daily_run"})
VALID_RECEIPT_VALIDATION_GATES = frozenset({"10/10"})
VALID_RECEIPT_PIPELINE_CONCLUSIONS = frozenset({"success"})


def _read_manifest(path):
    """Lee el manifest. Devuelve dict o None si no legible."""
    p = PROJECT_ROOT / path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_satisfies(manifest, target_session):
    """True si el manifest cubre target_session con cobertura suficiente.

    2026-09-29: verifica sha256 del parquet real. El manifest puede
    venir de CI (tracked) mientras el parquet local es de otro run
    (gitignored). Sin esto el gate decia CURRENT sobre un parquet
    que no corresponde al manifest.
    """
    if not isinstance(manifest, dict):
        return False
    q = manifest.get("quality")
    if not isinstance(q, dict):
        return False
    if q.get("expected_session") != target_session:
        return False
    cov = q.get("coverage_pct_last")
    if not isinstance(cov, (int, float)):
        return False
    if cov < MANIFEST_THRESHOLD:
        return False
    declared_sha = (manifest.get("artifact") or {}).get("sha256")
    if not isinstance(declared_sha, str) or not declared_sha:
        return False
    p = PROJECT_ROOT / MANIFEST_PATH.replace(".manifest.json", "")
    if not p.exists():
        return False
    return _sha256_file(p).lower() == declared_sha.lower()


def _probe_panel_once(target_session, tickers):
    """Descarga fresca del panel. Devuelve (coverage, error).

    coverage: fraccion de tickers con Close no-NaN en target_session.
    error: None si OK, string si fallo tecnico del probe.

    Un retorno (0.0, None) significa "Yahoo aun no tiene el dato" y
    se trata como NOT_READY, no como ERROR. Solo excepciones reales
    (red, pandas) se reportan como error.
    """
    session = None
    try:
        from curl_cffi import requests as curl_requests
        session = curl_requests.Session(impersonate="chrome")
    except Exception:
        pass

    ts = pd.Timestamp(target_session)
    start = ts.strftime("%Y-%m-%d")
    end = (ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        kwargs = {"start": start, "end": end, "progress": False,
                  "auto_adjust": False}
        if session is not None:
            kwargs["session"] = session
        data = yf.download(list(tickers), **kwargs)
    except Exception as e:
        return 0.0, "probe failed: {0}".format(e)

    if data is None or data.empty:
        return 0.0, None

    try:
        if isinstance(data.columns, pd.MultiIndex):
            close_block = data["Close"]
        else:
            if "Close" in data.columns:
                close_block = data[["Close"]]
            else:
                return 0.0, None
    except Exception as e:
        return 0.0, "probe column extraction failed: {0}".format(e)

    if close_block is None or close_block.empty:
        return 0.0, None

    if ts not in close_block.index:
        return 0.0, None

    row = close_block.loc[ts]
    n_total = len(tickers)
    n_valid = int(row.notna().sum())
    return n_valid / n_total, None


def _probe_panel(target_session, tickers=GATE_PANEL_USA):
    """Probe con retry corto para errores transitorios de red.

    Retry SOLO si _probe_panel_once devuelve error (excepcion de red).
    Si devuelve (0.0, None) es NOT_READY (Yahoo aun sin Close), sin retry.
    Tras agotar los intentos, devuelve (0.0, last_error) -> ERROR.
    """
    last_err = None
    for attempt in range(1, PROBE_MAX_RETRIES + 1):
        cov, err = _probe_panel_once(target_session, tickers)
        if err is None:
            return cov, None
        last_err = err
        if attempt < PROBE_MAX_RETRIES:
            time.sleep(PROBE_RETRY_SLEEPS[attempt - 1])
    return 0.0, last_err


def resolve_slot_flags(slot_expr):
    """Devuelve (is_known_slot, is_last_slot).

    slot_expr vacio o "manual" -> (False, False).
    slot_expr no reconocido -> (False, False).
    """
    if not slot_expr or slot_expr == "manual":
        return False, False
    if slot_expr not in CRON_SLOTS:
        return False, False
    return True, slot_expr == LAST_SLOT_CRON


# ---------------- Completion Receipt (contrato v1) ----------------


def _github_token():
    """Token de GitHub desde GH_TOKEN o GITHUB_TOKEN. None si no hay."""
    return os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")


def _github_repo():
    """owner/name del repo. Env o default."""
    return os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO)


def _github_api(path, params=None):
    """GET a la API de GitHub. Devuelve dict o None si falla.

    Cumple I6: cualquier error (red, 401/403/404/5xx, JSON invalido)
    devuelve None. Nunca declara CURRENT por fallo de API.
    """
    repo = _github_repo()
    url = "https://api.github.com/repos/{0}/{1}".format(repo, path)
    headers = {"Accept": "application/vnd.github+json"}
    token = _github_token()
    if token:
        headers["Authorization"] = "Bearer {0}".format(token)
    try:
        r = requests.get(url, headers=headers, params=params,
                         timeout=GH_API_TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _download_receipt_json(artifact_id):
    """Descarga el ZIP del artifact y extrae receipt.json.

    Devuelve dict o None. Requiere token (el ZIP endpoint lo exige).
    """
    token = _github_token()
    if not token:
        return None
    repo = _github_repo()
    url = ("https://api.github.com/repos/{0}/actions/artifacts/"
           "{1}/zip").format(repo, artifact_id)
    headers = {
        "Authorization": "Bearer {0}".format(token),
        "Accept": "application/vnd.github+json",
    }
    try:
        r = requests.get(url, headers=headers,
                         timeout=GH_API_ZIP_TIMEOUT, allow_redirects=True)
        if r.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            if RECEIPT_FILENAME not in z.namelist():
                return None
            data = z.read(RECEIPT_FILENAME)
        return json.loads(data)
    except Exception:
        return None


def _validate_receipt_schema(receipt, target_session):
    """True si el receipt cumple el schema del contrato v1, seccion 3."""
    if not isinstance(receipt, dict):
        return False
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        return False
    if receipt.get("status") not in VALID_RECEIPT_STATUS:
        return False
    if receipt.get("target_session") != target_session:
        return False
    if receipt.get("workflow") not in VALID_RECEIPT_WORKFLOWS:
        return False
    if not isinstance(receipt.get("run_id"), int):
        return False
    if receipt.get("validation_gate") not in VALID_RECEIPT_VALIDATION_GATES:
        return False
    if receipt.get("pipeline_conclusion") not in VALID_RECEIPT_PIPELINE_CONCLUSIONS:
        return False
    if not isinstance(receipt.get("completed_at"), str):
        return False
    if not receipt.get("completed_at"):
        return False
    return True


def find_completion_receipt(target_session):
    """Devuelve el receipt validado o None.

    Capa 1: existe artifact `completion-receipt-<target_session>` y su
            contenido cumple el schema del contrato v1.
    Capa 2: el workflow run que lo subio tiene conclusion == "success".

    Cualquier fallo (sin artifact, schema invalido, run no exitoso,
    error de API, sin token) devuelve None. Cumple I6. Toda excepcion
    no capturada internamente se captura aqui para garantizar que
    jamas escapa un error a evaluate().
    """
    try:
        return _find_completion_receipt_impl(target_session)
    except Exception:
        return None


def _find_completion_receipt_impl(target_session):
    """Implementacion de find_completion_receipt. Ver docstring publico."""
    artifact_name = "{0}{1}".format(RECEIPT_ARTIFACT_PREFIX, target_session)

    data = _github_api("actions/artifacts", params={"name": artifact_name})
    if not data:
        return None
    artifacts = data.get("artifacts") or []
    if not artifacts:
        return None

    # El mas reciente por created_at (evita recibir uno antiguo si hay varios)
    artifacts_sorted = sorted(
        artifacts,
        key=lambda a: a.get("created_at") or "",
        reverse=True,
    )
    artifact = artifacts_sorted[0]

    artifact_id = artifact.get("id")
    if not isinstance(artifact_id, int):
        return None

    workflow_run = artifact.get("workflow_run") or {}
    run_id = workflow_run.get("id")
    if not isinstance(run_id, int):
        return None

    # Capa 2: el run que genero el receipt debe haber terminado OK.
    run_data = _github_api("actions/runs/{0}".format(run_id))
    if not run_data:
        return None
    if run_data.get("conclusion") != "success":
        return None

    receipt = _download_receipt_json(artifact_id)
    if receipt is None:
        return None

    if not _validate_receipt_schema(receipt, target_session):
        return None

    return receipt


# ---------------- Gate ----------------


def evaluate(target_session):
    """Determina el estado del gate para un target_session dado."""
    receipt = find_completion_receipt(target_session)
    if receipt is not None:
        return {
            "state": "CURRENT",
            "should_run": False,
            "reason": "completion receipt for {0} (run {1})".format(
                target_session, receipt["run_id"]),
            "expected_session": target_session,
            "probe_coverage": None,
            "manifest_coverage": None,
            "receipt_run_id": receipt["run_id"],
        }

    probe_cov, probe_err = _probe_panel(target_session)
    if probe_err is not None:
        return {
            "state": "ERROR",
            "should_run": False,
            "reason": probe_err,
            "expected_session": target_session,
            "probe_coverage": None,
            "manifest_coverage": None,
        }

    if probe_cov >= PROBE_MIN_COVERAGE:
        return {
            "state": "READY",
            "should_run": True,
            "reason": "probe coverage {0:.2%} >= {1:.0%}".format(
                probe_cov, PROBE_MIN_COVERAGE),
            "expected_session": target_session,
            "probe_coverage": probe_cov,
            "manifest_coverage": None,
        }

    return {
        "state": "NOT_READY",
        "should_run": False,
        "reason": "probe coverage {0:.2%} < {1:.0%}".format(
            probe_cov, PROBE_MIN_COVERAGE),
        "expected_session": target_session,
        "probe_coverage": probe_cov,
        "manifest_coverage": None,
    }


def _write_github_output(result):
    """Escribe outputs al GITHUB_OUTPUT si esta disponible."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    try:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("state={0}\n".format(result["state"]))
            f.write("should_run={0}\n".format(
                "true" if result["should_run"] else "false"))
            f.write("expected_session={0}\n".format(result["expected_session"]))
            f.write("reason={0}\n".format(result["reason"]))
            f.write("is_known_slot={0}\n".format(
                "true" if result.get("is_known_slot") else "false"))
            f.write("is_last_slot={0}\n".format(
                "true" if result.get("is_last_slot") else "false"))
    except Exception:
        pass


def resolve_target_session(now=None):
    """Resuelve target_session a partir de now (default: UTC)."""
    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    return last_expected_market_date(now).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", default=None,
                    help="ISO datetime (testing). Default: now UTC")
    ap.add_argument("--target-session", default=None,
                    help="Forzar target_session ISO date (testing)")
    ap.add_argument("--slot", default="",
                    help="github.event.schedule (cron string). "
                         "Vacio o manual en workflow_dispatch.")
    args = ap.parse_args()

    if args.target_session:
        target = args.target_session
    elif args.now:
        target = resolve_target_session(datetime.fromisoformat(args.now))
    else:
        target = resolve_target_session()

    result = evaluate(target)
    result["is_known_slot"], result["is_last_slot"] = resolve_slot_flags(args.slot)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    _write_github_output(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
