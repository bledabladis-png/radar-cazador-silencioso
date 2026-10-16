"""Tests HD-5: separacion Schedule vs Dispatch en health_check.

Dictamen auditor externo 2026-10-16: un workflow_dispatch reciente
no debe ocultar que el schedule automatico no esta disparando. El
check debe diferenciar las dos dimensiones.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.health_check as hc


def _iso(hours_ago):
    t = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(event, hours_ago, conclusion="success"):
    return {
        "createdAt": _iso(hours_ago),
        "conclusion": conclusion,
        "status": "completed",
        "event": event,
    }


def _mock_gh(sched, any_):
    def fake(args):
        if "--event" in args and args[args.index("--event") + 1] == "schedule":
            return json.dumps(sched)
        return json.dumps(any_)
    return fake


def _check(meta, sched, any_):
    with patch.object(hc, "_run_gh", side_effect=_mock_gh(sched, any_)):
        return hc._check_one_workflow("daily_run.yml", meta)


META_DIARIO = {"max_days": 2}
META_TRIM = {"max_days": 100}


def test_schedule_ok():
    r = _check(META_DIARIO,
               [_run("schedule", 1)],
               [_run("schedule", 1)])
    assert r.status == hc.OK


def test_schedule_warn_failure():
    r = _check(META_DIARIO,
               [_run("schedule", 1, conclusion="failure")],
               [_run("schedule", 1, conclusion="failure")])
    assert r.status == hc.WARN


def test_schedule_fail_stale_no_dispatch():
    """Schedule stale sin dispatch -> FAIL (no trimestral)."""
    r = _check(META_DIARIO,
               [_run("schedule", 24 * 10)],
               [_run("schedule", 24 * 10)])
    assert r.status == hc.FAIL


def test_dispatch_only_warn():
    """HD-5: schedule vacio + dispatch reciente -> WARN, NO OK."""
    r = _check(META_DIARIO,
               [],
               [_run("workflow_dispatch", 1)])
    assert r.status == hc.WARN
    assert "SCHEDULE" in r.message
    assert "workflow_dispatch" in r.message


def test_schedule_stale_dispatch_recent_warn():
    """HD-5: schedule stale + dispatch reciente -> WARN, NO FAIL."""
    r = _check(META_DIARIO,
               [_run("schedule", 24 * 10)],
               [_run("workflow_dispatch", 1)])
    assert r.status == hc.WARN
    assert "SCHEDULE stale" in r.message
    assert "DISPATCH reciente" in r.message


def test_trimestral_skip_sin_schedule():
    """Trimestral sin schedule pero con dispatch -> SKIP (esperado)."""
    r = _check(META_TRIM,
               [],
               [_run("workflow_dispatch", 100)])
    assert r.status == hc.SKIP


def test_trimestral_skip_sin_nada():
    r = _check(META_TRIM, [], [])
    assert r.status == hc.SKIP


def test_ningun_run_diario_warn():
    """Diario sin ningun run -> WARN, no OK."""
    r = _check(META_DIARIO, [], [])
    assert r.status == hc.WARN
    assert "sin ejecuciones" in r.message
