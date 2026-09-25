#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate de disponibilidad para el pipeline diario.

Estados (dictamen auditor 2026-09-25, C3):
  CURRENT    Manifest ya cumple cobertura para target_session. No correr.
  READY      Manifest no cumple, probe OK. Correr pipeline.
  NOT_READY  Manifest no cumple, probe por debajo del threshold. Skip.
  ERROR      Fallo tecnico del gate. Skip.

Concepto clave (C7): target_session es la sesion bursatil que el
pipeline intenta producir, NO el dia calendario UTC del slot.

Exit code: siempre 0. La decision viaja por outputs de GitHub.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
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
    "17 7 * * *",
    "17 11 * * *",
)
LAST_SLOT_CRON = "17 11 * * *"


def _read_manifest(path):
    """Lee el manifest. Devuelve dict o None si no legible."""
    p = PROJECT_ROOT / path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _manifest_satisfies(manifest, target_session):
    """True si el manifest ya cubre target_session con cobertura suficiente."""
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
    return cov >= MANIFEST_THRESHOLD


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


def evaluate(target_session):
    """Determina el estado del gate para un target_session dado."""
    manifest = _read_manifest(MANIFEST_PATH)
    if _manifest_satisfies(manifest, target_session):
        return {
            "state": "CURRENT",
            "should_run": False,
            "reason": "manifest already covers {0}".format(target_session),
            "expected_session": target_session,
            "probe_coverage": None,
            "manifest_coverage": manifest["quality"]["coverage_pct_last"],
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
