# -*- coding: utf-8 -*-
"""Test: verifica_gobernanza rechaza SHA huerfanos en items FIXED.

Item: REP-AUDIT-20261022-GOVERNANCE-SHA-REACHABILITY
Causa: git cat-file -e acepta objetos huerfanos (post-rebase, aun
no recolectados). Un item FIXED con SHA huerfano apunta a un commit
que desaparecera al correr git GC.
"""

import subprocess

import scripts.verifica_gobernanza as vg


def _run_git(*args, cwd=None):
    return subprocess.run(
        ["git"] + list(args),
        cwd=str(cwd) if cwd else None,
        capture_output=True, text=True,
    )


def test_valid_sha_passes():
    """HEAD actual debe existir y ser alcanzable."""
    r = _run_git("rev-parse", "HEAD", cwd=vg.ROOT)
    head = r.stdout.strip()
    assert head, "no se pudo obtener HEAD"
    assert vg._git_commit_exists(head) is True


def test_missing_sha_fails():
    """SHA inventado que no existe en la BD de objetos."""
    fake = "0" * 40
    assert vg._git_commit_exists(fake) is False


def test_pending_returns_none():
    """PENDING, vacio o None no verifican."""
    assert vg._git_commit_exists("PENDING") is None
    assert vg._git_commit_exists("") is None
    assert vg._git_commit_exists(None) is None


def test_orphan_sha_fails(tmp_path, monkeypatch):
    """SHA que existe como objeto pero no es alcanzable desde refs."""
    _run_git("init", str(tmp_path))
    _run_git("config", "user.email", "test@example.com", cwd=tmp_path)
    _run_git("config", "user.name", "Test", cwd=tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    _run_git("add", "a.txt", cwd=tmp_path)
    _run_git("commit", "-m", "init", cwd=tmp_path)

    r = _run_git("rev-parse", "HEAD^{tree}", cwd=tmp_path)
    tree = r.stdout.strip()
    assert tree, "no se pudo obtener tree"
    r = _run_git("commit-tree", tree, "-m", "orphan", cwd=tmp_path)
    orphan = r.stdout.strip()
    assert orphan, "no se pudo crear commit huerfano"

    r = _run_git("cat-file", "-e", orphan, cwd=tmp_path)
    assert r.returncode == 0, "el objeto huerfano deberia existir"

    monkeypatch.setattr(vg, "ROOT", tmp_path)

    assert vg._git_commit_exists(orphan) is False