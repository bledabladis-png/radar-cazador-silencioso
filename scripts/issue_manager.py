#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue manager para el pipeline diario.

Se ejecuta tras `gate` y `run-system`. Decide si abrir, comentar, cerrar
o no hacer nada con el Issue de disponibilidad de una target_session.

Diseno (dictamen auditor 2026-09-25):

  decide_action() es PURA: solo depende de 4 parametros escalares.
  validate_target_session() es PURA y previa.
  execute_action() hace la I/O via gh CLI.

Contrato de estados (tabla aprobada):
  CURRENT                      -> CLOSE (recuperacion confirmada)
  READY + success              -> CLOSE
  READY + failure/cancelled    -> NOOP  (mantener abierto)
  READY + skipped              -> NOOP
  NOT_READY + last_slot + sch  -> ENSURE_FAILURE
  NOT_READY + non-last + sch   -> NOOP
  ERROR + last_slot + sch      -> ENSURE_FAILURE
  ERROR + non-last + sch       -> NOOP
  cualquier + workflow_dispatch:
      CLOSE solo si CURRENT o (READY + success)
      nunca ENSURE_FAILURE

Uso: invocado por `daily_run.yml` (job issue-manager). Lee env vars.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date

LABEL = "daily-run-failure"
TITLE_TEMPLATE = "Daily data availability failure - {session}"

VALID_GATE_STATES = frozenset({"CURRENT", "READY", "NOT_READY", "ERROR"})
VALID_RUN_SYSTEM_RESULTS = frozenset(
    {"success", "failure", "cancelled", "skipped"}
)
VALID_EVENTS = frozenset({"schedule", "workflow_dispatch"})

ACTION_NOOP = "NOOP"
ACTION_CLOSE = "CLOSE"
ACTION_ENSURE_FAILURE = "ENSURE_FAILURE"


def validate_target_session(s):
    """True si s es una fecha ISO YYYY-MM-DD valida y no vacia."""
    if not isinstance(s, str) or not s:
        return False
    try:
        date.fromisoformat(s)
    except (ValueError, TypeError):
        return False
    return True


def decide_action(gate_state, run_system_result, is_last_slot, event_name):
    """Decide NOOP | CLOSE | ENSURE_FAILURE.

    Funcion pura. No conoce si el Issue existe. La existencia se
    resuelve en execute_action (ENSURE_FAILURE busca y crea/comenta).
    """
    if gate_state not in VALID_GATE_STATES:
        return ACTION_NOOP
    if event_name not in VALID_EVENTS:
        return ACTION_NOOP

    # CLOSE: recuperacion confirmada por gate o pipeline.
    if gate_state == "CURRENT":
        return ACTION_CLOSE
    if gate_state == "READY" and run_system_result == "success":
        return ACTION_CLOSE

    # ENSURE_FAILURE: solo en schedule, solo en ultimo slot, solo en
    # estados que representan ausencia real de datos.
    if event_name == "schedule" and is_last_slot:
        if gate_state in ("NOT_READY", "ERROR"):
            return ACTION_ENSURE_FAILURE

    return ACTION_NOOP


def _run_gh(args):
    """Ejecuta gh CLI. Devuelve stdout o None si fallo.

    Patron copiado de scripts/health_check.py::_run_gh.
    """
    try:
        r = subprocess.run(
            ["gh"] + args, capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            return None
        return r.stdout.strip()
    except Exception:
        return None


def _find_issue_by_title(target_session):
    """Devuelve el numero del Issue abierto con label + title exacto.

    Comparacion EXACTA de title en Python (no confiar en --search).
    """
    expected_title = TITLE_TEMPLATE.format(session=target_session)
    out = _run_gh([
        "issue", "list",
        "--label", LABEL,
        "--state", "open",
        "--json", "number,title",
        "--limit", "100",
    ])
    if out is None:
        return None
    try:
        items = json.loads(out)
    except Exception:
        return None
    for it in items:
        if it.get("title") == expected_title:
            return it.get("number")
    return None


def _ensure_label():
    """Crea el label si no existe. Idempotente (--force)."""
    subprocess.run(
        ["gh", "label", "create", LABEL, "--force",
         "--description", "Daily data availability failure",
         "--color", "d93f0b"],
        capture_output=True, text=True, timeout=15,
    )


def execute_action(action, *, target_session, reason, run_id):
    """Ejecuta la accion via gh CLI. No-op si action == NOOP."""
    if action == ACTION_NOOP:
        return

    if not validate_target_session(target_session):
        print("[issue-manager] target_session invalido, NOOP")
        return

    title = TITLE_TEMPLATE.format(session=target_session)
    run_url = "https://github.com/{0}/actions/runs/{1}".format(
        os.environ.get("GITHUB_REPOSITORY", "bledabladis-png/radar-cazador-silencioso"),
        run_id,
    )

    if action == ACTION_CLOSE:
        existing = _find_issue_by_title(target_session)
        if existing is None:
            print("[issue-manager] CLOSE: no hay issue abierto para "
                  + target_session)
            return
        _run_gh([
            "issue", "close", str(existing),
            "--comment",
            "Recuperado. target_session={0}. Run: {1}".format(
                target_session, run_url),
        ])
        print("[issue-manager] issue #{0} cerrado".format(existing))
        return

    if action == ACTION_ENSURE_FAILURE:
        _ensure_label()
        body = (
            "Disponibilidad de datos no confirmada.\n\n"
            "- target_session: {0}\n"
            "- gate state: {1}\n"
            "- reason: {2}\n"
            "- run: {3}\n".format(
                target_session,
                os.environ.get("GATE_STATE", "?"),
                reason,
                run_url,
            )
        )
        existing = _find_issue_by_title(target_session)
        if existing is not None:
            _run_gh(["issue", "comment", str(existing), "--body", body])
            print("[issue-manager] comentado en issue #{0}".format(existing))
        else:
            r = subprocess.run(
                ["gh", "issue", "create",
                 "--title", title,
                 "--body", body,
                 "--label", LABEL],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                print("[issue-manager] issue creado: {0}".format(
                    r.stdout.strip()))
            else:
                print("[issue-manager] fallo creando issue: {0}".format(
                    r.stderr.strip()))


def main():
    gate_state = os.environ.get("GATE_STATE", "")
    run_system_result = os.environ.get("RUN_SYSTEM_RESULT", "")
    is_last_slot = os.environ.get("IS_LAST_SLOT", "false").lower() == "true"
    event_name = os.environ.get("EVENT_NAME", "")
    target_session = os.environ.get("TARGET_SESSION", "")
    reason = os.environ.get("GATE_REASON", "")
    run_id = os.environ.get("RUN_ID", "")

    action = decide_action(
        gate_state, run_system_result, is_last_slot, event_name
    )
    print("[issue-manager] gate={0} run_system={1} last={2} event={3} "
          "-> action={4}".format(
              gate_state, run_system_result, is_last_slot,
              event_name, action))

    execute_action(
        action,
        target_session=target_session,
        reason=reason,
        run_id=run_id,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
