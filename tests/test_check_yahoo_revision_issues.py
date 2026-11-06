# -*- coding: utf-8 -*-
"""Tests de apertura/cierre de issues en check_yahoo_revision (Q10, S30)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_yahoo_revision as cyr  # noqa: E402


class FakeResult:
    def __init__(self, name, status, message):
        self.name = name
        self.status = status
        self.message = message


def test_aggregate_ok():
    assert cyr._aggregate_status([FakeResult("y", cyr.OK, "fine")]) == cyr.OK


def test_aggregate_warn():
    assert cyr._aggregate_status([FakeResult("y", cyr.WARN, "rev")]) == cyr.WARN


def test_aggregate_fail_prevalece():
    rs = [FakeResult("y", cyr.WARN, "rev"), FakeResult("y", cyr.FAIL, "boom")]
    assert cyr._aggregate_status(rs) == cyr.FAIL


def test_aggregate_skip():
    assert cyr._aggregate_status([FakeResult("y", cyr.SKIP, "no manifest")]) == cyr.SKIP


def test_update_ok_sin_issue_no_hace_nada(monkeypatch):
    calls = []
    monkeypatch.setattr(cyr, "_ensure_label", lambda: calls.append("label"))
    monkeypatch.setattr(cyr, "_find_open_issue", lambda: None)
    monkeypatch.setattr(cyr, "_run_gh", lambda args: calls.append(("gh", tuple(args))) or "")
    cyr._update_issue([FakeResult("y", cyr.OK, "fine")])
    assert calls == ["label"]


def test_update_ok_con_issue_cierra(monkeypatch):
    gh_calls = []
    monkeypatch.setattr(cyr, "_ensure_label", lambda: None)
    monkeypatch.setattr(cyr, "_find_open_issue", lambda: 42)
    def fake_gh(args):
        gh_calls.append(tuple(args))
        return ""
    monkeypatch.setattr(cyr, "_run_gh", fake_gh)
    cyr._update_issue([FakeResult("y", cyr.OK, "fine")])
    assert any("close" in c for c in gh_calls)


def test_update_warn_sin_issue_crea(monkeypatch):
    subprocess_calls = []
    monkeypatch.setattr(cyr, "_ensure_label", lambda: None)
    monkeypatch.setattr(cyr, "_find_open_issue", lambda: None)
    monkeypatch.setattr(cyr, "_run_gh", lambda args: "")
    class FakeProc:
        returncode = 0
        stdout = "https://github.com/x/issues/1"
        stderr = ""
    def fake_run(cmd, **kw):
        subprocess_calls.append(cmd)
        return FakeProc()
    monkeypatch.setattr(cyr.subprocess, "run", fake_run)
    cyr._update_issue([FakeResult("y", cyr.WARN, "revision detectada")])
    assert any("issue" in c and "create" in c for c in subprocess_calls)


def test_update_warn_con_issue_actualiza(monkeypatch):
    gh_calls = []
    monkeypatch.setattr(cyr, "_ensure_label", lambda: None)
    monkeypatch.setattr(cyr, "_find_open_issue", lambda: 42)
    def fake_gh(args):
        gh_calls.append(tuple(args))
        return ""
    monkeypatch.setattr(cyr, "_run_gh", fake_gh)
    cyr._update_issue([FakeResult("y", cyr.WARN, "revision")])
    assert any("comment" in c for c in gh_calls)
    assert any("edit" in c for c in gh_calls)


def test_update_skip_no_hace_nada(monkeypatch):
    calls = []
    monkeypatch.setattr(cyr, "_ensure_label", lambda: calls.append("label"))
    cyr._update_issue([FakeResult("y", cyr.SKIP, "no manifest")])
    assert calls == []


def test_main_sin_flag_no_toca_issues(monkeypatch):
    touched = []
    monkeypatch.setattr(cyr, "check_yahoo_revision",
                        lambda: [FakeResult("y", cyr.WARN, "rev")])
    monkeypatch.setattr(cyr, "_update_issue", lambda rs: touched.append(rs))
    cyr.main([])
    assert touched == []


def test_main_con_flag_llama_update_issue(monkeypatch):
    touched = []
    monkeypatch.setattr(cyr, "check_yahoo_revision",
                        lambda: [FakeResult("y", cyr.WARN, "rev")])
    monkeypatch.setattr(cyr, "_update_issue", lambda rs: touched.append(rs))
    cyr.main(["--with-issues"])
    assert len(touched) == 1


def test_main_devuelve_cero_ante_excepcion_de_issues(monkeypatch):
    monkeypatch.setattr(cyr, "check_yahoo_revision",
                        lambda: [FakeResult("y", cyr.WARN, "rev")])
    def boom(rs):
        raise RuntimeError("gh roto")
    monkeypatch.setattr(cyr, "_update_issue", boom)
    assert cyr.main(["--with-issues"]) == 0
