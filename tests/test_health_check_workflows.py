"""check_workflows: mira el ultimo run de cualquier evento, no solo schedule.

Regresion 2026-10-08: los workflows trimestrales tuvieron schedule
failure el 1-oct-2026, corregidos ese mismo dia. Los dispatch
posteriores pasan. El check original filtraba --event schedule y
mostraba WARN permanente.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from scripts.health_check import check_workflows


def _mock_gh(runs_json):
    def fake(cmd):
        return json.dumps(runs_json)
    return fake


def _iso_ago(days):
    t = datetime.now(timezone.utc) - timedelta(days=days)
    return t.isoformat().replace("+00:00", "Z")


def test_dispatch_reciente_ok():
    runs = [{"createdAt": _iso_ago(0.1), "conclusion": "success",
             "status": "completed", "event": "workflow_dispatch"}]
    with patch("scripts.health_check._run_gh", _mock_gh(runs)):
        results = check_workflows()
    daily = [r for r in results if r.name == "workflow:daily_run.yml"][0]
    assert daily.status == "OK", daily


def test_schedule_failure_reciente_es_warn():
    runs = [{"createdAt": _iso_ago(1), "conclusion": "failure",
             "status": "completed", "event": "schedule"}]
    with patch("scripts.health_check._run_gh", _mock_gh(runs)):
        results = check_workflows()
    daily = [r for r in results if r.name == "workflow:daily_run.yml"][0]
    assert daily.status == "WARN", daily


def test_run_viejo_es_fail():
    runs = [{"createdAt": _iso_ago(10), "conclusion": "success",
             "status": "completed", "event": "workflow_dispatch"}]
    with patch("scripts.health_check._run_gh", _mock_gh(runs)):
        results = check_workflows()
    daily = [r for r in results if r.name == "workflow:daily_run.yml"][0]
    assert daily.status == "FAIL", daily


def test_trimestral_sin_runs_es_skip():
    with patch("scripts.health_check._run_gh", _mock_gh([])):
        results = check_workflows()
    sector = [r for r in results if r.name == "workflow:update_sector_holdings.yml"][0]
    assert sector.status == "SKIP", sector

def test_cancelled_es_warn():
    """Un run cancelled no es exito. No debe marcar OK."""
    runs = [{"createdAt": _iso_ago(1), "conclusion": "cancelled",
             "status": "completed", "event": "workflow_dispatch"}]
    with patch("scripts.health_check._run_gh", _mock_gh(runs)):
        results = check_workflows()
    daily = [r for r in results if r.name == "workflow:daily_run.yml"][0]
    assert daily.status == "WARN", daily
    assert "cancelled" in daily.message
