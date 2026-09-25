# -*- coding: utf-8 -*-
"""Tests del issue_manager (F-IAE-CRON-02/2b-1).

Cubre:
  - validate_target_session (puro).
  - decide_action (puro, 3 acciones x tabla de 10 ramas).
  - execute_action (subprocess.run mockeado).
  - Contrato de seguridad (GH_TOKEN no se filtra).
  - Drift test: CRON_SLOTS en pipeline_gate.py vs daily_run.yml.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.issue_manager as im
from scripts.issue_manager import (
    ACTION_NOOP,
    ACTION_CLOSE,
    ACTION_ENSURE_FAILURE,
    decide_action,
    validate_target_session,
)


# ---------------- validate_target_session ----------------

def test_validate_ok():
    assert validate_target_session("2026-09-24") is True


def test_validate_vacio():
    assert validate_target_session("") is False


def test_validate_none():
    assert validate_target_session(None) is False


def test_validate_no_iso():
    assert validate_target_session("24/09/2026") is False


def test_validate_iso_invalido():
    assert validate_target_session("2026-13-01") is False


def test_validate_int():
    assert validate_target_session(20260924) is False


# ---------------- decide_action ----------------

def test_current_close():
    assert decide_action("CURRENT", "skipped", False, "schedule") == ACTION_CLOSE


def test_ready_success_close():
    assert decide_action("READY", "success", False, "schedule") == ACTION_CLOSE


def test_ready_failure_noop():
    assert decide_action("READY", "failure", False, "schedule") == ACTION_NOOP


def test_ready_cancelled_noop():
    assert decide_action("READY", "cancelled", False, "schedule") == ACTION_NOOP


def test_ready_skipped_noop():
    assert decide_action("READY", "skipped", False, "schedule") == ACTION_NOOP


def test_not_ready_last_slot_ensure():
    assert decide_action("NOT_READY", "skipped", True, "schedule") == ACTION_ENSURE_FAILURE


def test_not_ready_non_last_noop():
    assert decide_action("NOT_READY", "skipped", False, "schedule") == ACTION_NOOP


def test_error_last_slot_ensure():
    assert decide_action("ERROR", "skipped", True, "schedule") == ACTION_ENSURE_FAILURE


def test_error_non_last_noop():
    assert decide_action("ERROR", "skipped", False, "schedule") == ACTION_NOOP


def test_dispatch_current_close():
    """workflow_dispatch + CURRENT -> CLOSE (recuperacion manual)."""
    assert decide_action("CURRENT", "skipped", False, "workflow_dispatch") == ACTION_CLOSE


def test_dispatch_ready_success_close():
    assert decide_action("READY", "success", False, "workflow_dispatch") == ACTION_CLOSE


def test_dispatch_not_ready_last_noop():
    """workflow_dispatch nunca abre issue."""
    assert decide_action("NOT_READY", "skipped", True, "workflow_dispatch") == ACTION_NOOP


def test_dispatch_error_last_noop():
    assert decide_action("ERROR", "skipped", True, "workflow_dispatch") == ACTION_NOOP


def test_gate_state_invalido_noop():
    assert decide_action("GARBAGE", "success", True, "schedule") == ACTION_NOOP


def test_event_invalido_noop():
    assert decide_action("NOT_READY", "skipped", True, "push") == ACTION_NOOP


# ---------------- execute_action (gh mockeado) ----------------

def _mk_subprocess_result(stdout="", stderr="", returncode=0):
    r = MagicMock()
    r.stdout = stdout
    r.stderr = stderr
    r.returncode = returncode
    return r


def test_noop_no_invoca_gh():
    with patch.object(im.subprocess, "run") as m:
        im.execute_action(ACTION_NOOP, target_session="2026-09-24",
                          reason="x", run_id="123")
    assert m.call_count == 0


def test_close_sin_issue_existente_noop():
    """CLOSE sin issue abierto -> no invoca close."""
    with patch.object(im, "_find_issue_by_title", return_value=None), \
         patch.object(im, "_run_gh") as m_run:
        im.execute_action(ACTION_CLOSE, target_session="2026-09-24",
                          reason="ok", run_id="123")
    m_run.assert_not_called()


def test_close_con_issue_existente_cierra():
    with patch.object(im, "_find_issue_by_title", return_value=42), \
         patch.object(im, "_run_gh") as m_run:
        im.execute_action(ACTION_CLOSE, target_session="2026-09-24",
                          reason="ok", run_id="123")
    assert m_run.call_count == 1
    args = m_run.call_args[0][0]
    assert args[0] == "issue"
    assert args[1] == "close"
    assert args[2] == "42"


def test_close_target_session_invalido_noop():
    with patch.object(im, "_find_issue_by_title") as m_find:
        im.execute_action(ACTION_CLOSE, target_session="",
                          reason="ok", run_id="123")
    m_find.assert_not_called()


def test_ensure_failure_crea_issue_si_no_existe():
    with patch.object(im, "_ensure_label"), \
         patch.object(im, "_find_issue_by_title", return_value=None), \
         patch.object(im.subprocess, "run") as m_sub:
        m_sub.return_value = _mk_subprocess_result(
            stdout="https://github.com/x/y/issues/1"
        )
        im.execute_action(ACTION_ENSURE_FAILURE, target_session="2026-09-24",
                          reason="not ready", run_id="123")
    assert m_sub.call_count == 1
    args = m_sub.call_args[0][0]
    assert args[:3] == ["gh", "issue", "create"]
    assert "--title" in args
    assert "--label" in args


def test_ensure_failure_comenta_si_existe():
    with patch.object(im, "_ensure_label"), \
         patch.object(im, "_find_issue_by_title", return_value=7), \
         patch.object(im, "_run_gh") as m_run, \
         patch.object(im.subprocess, "run") as m_sub:
        im.execute_action(ACTION_ENSURE_FAILURE, target_session="2026-09-24",
                          reason="not ready", run_id="123")
    assert m_run.call_count == 1
    args = m_run.call_args[0][0]
    assert args[:3] == ["issue", "comment", "7"]
    m_sub.assert_not_called()


def test_ensure_failure_target_session_invalido_noop():
    with patch.object(im, "_ensure_label") as m_label, \
         patch.object(im, "_find_issue_by_title") as m_find:
        im.execute_action(ACTION_ENSURE_FAILURE, target_session="bad",
                          reason="x", run_id="123")
    m_label.assert_not_called()
    m_find.assert_not_called()


def test_find_issue_comparacion_exacta_de_title():
    """Match exacto de title, no 'startswith' ni 'in'."""
    payload = (
        '[{"number": 1, "title": "Daily data availability failure - 2026-09-25"},'
        ' {"number": 2, "title": "Daily data availability failure - 2026-09-24"},'
        ' {"number": 3, "title": "Other issue"}]'
    )
    with patch.object(im, "_run_gh", return_value=payload):
        assert im._find_issue_by_title("2026-09-24") == 2


def test_find_issue_ninguno_coincide():
    payload = '[{"number": 1, "title": "Daily data availability failure - 2026-09-23"}]'
    with patch.object(im, "_run_gh", return_value=payload):
        assert im._find_issue_by_title("2026-09-24") is None


def test_find_issue_gh_falla_devuelve_none():
    with patch.object(im, "_run_gh", return_value=None):
        assert im._find_issue_by_title("2026-09-24") is None


# ---------------- seguridad ----------------

def test_gh_token_no_en_args():
    """GH_TOKEN nunca debe aparecer en argv de subprocess.run."""
    captured = []
    def fake_run(args, **kw):
        captured.append(args)
        return _mk_subprocess_result()
    with patch.object(im, "_ensure_label"), \
         patch.object(im, "_find_issue_by_title", return_value=None), \
         patch.object(im.subprocess, "run", side_effect=fake_run):
        im.execute_action(ACTION_ENSURE_FAILURE, target_session="2026-09-24",
                          reason="x", run_id="123")
    for args in captured:
        for a in args:
            assert "ghp_" not in a
            assert "ghs_" not in a
            assert a != "GH_TOKEN"


# ---------------- drift: CRON_SLOTS ----------------

def test_drift_cron_slots_en_daily_run_yml():
    """CRON_SLOTS en pipeline_gate.py debe coincidir con cron: en daily_run.yml."""
    from scripts.pipeline_gate import CRON_SLOTS
    yml = (ROOT / ".github" / "workflows" / "daily_run.yml").read_text(
        encoding="utf-8")
    for slot in CRON_SLOTS:
        assert "cron: '" + slot + "'" in yml, (
            "slot {0} no encontrado en daily_run.yml".format(slot))
