# -*- coding: utf-8 -*-
"""Tests de scripts/hooks/install.py y los checks (S30).

No instala en .git/hooks real: usa tmp_path. No corre pytest real:
monkeypatchea subprocess.run. No toca .git/hooks del repo.

Carga los modulos por path (importlib.spec_from_file_location) para
no necesitar scripts/hooks/__init__.py. Evita colisiones con otros
modulos llamados install/pre_commit_check.
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
HOOKS = ROOT / "scripts" / "hooks"


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


install_mod = _load("hooks_install_mod", HOOKS / "install.py")
pre_commit_mod = _load("pre_commit_mod", HOOKS / "pre_commit_check.py")
pre_push_mod = _load("pre_push_mod", HOOKS / "pre_push_check.py")


# ---- install ----


def test_install_escribe_stubs(tmp_path):
    acciones = install_mod.install(hooks_dir=tmp_path, force=True)
    assert len(acciones) == 2
    pc = tmp_path / "pre-commit"
    pp = tmp_path / "pre-push"
    assert pc.exists() and pp.exists()
    txt_pc = pc.read_text(encoding="utf-8")
    txt_pp = pp.read_text(encoding="utf-8")
    assert txt_pc.startswith("#!/bin/sh")
    assert txt_pp.startswith("#!/bin/sh")
    assert "pre_commit_check.py" in txt_pc
    assert "pre_push_check.py" in txt_pp
    assert "exec py" in txt_pc


def test_install_backup_si_existe_no_sample(tmp_path):
    previo = tmp_path / "pre-commit"
    previo.write_text("#!/bin/sh\\necho old\\n", encoding="utf-8")
    acciones = install_mod.install(hooks_dir=tmp_path, force=True)
    bak = tmp_path / "pre-commit.orig"
    assert bak.exists(), "No se creo backup .orig"
    assert "echo old" in bak.read_text(encoding="utf-8")
    # El backup aparece en la accion reportada
    pc_accion = [a for p, a in acciones if p.endswith("pre-commit")][0]
    assert ".orig" in pc_accion


def test_install_idempotente(tmp_path):
    install_mod.install(hooks_dir=tmp_path, force=True)
    txt1 = (tmp_path / "pre-commit").read_text(encoding="utf-8")
    install_mod.install(hooks_dir=tmp_path, force=True)
    txt2 = (tmp_path / "pre-commit").read_text(encoding="utf-8")
    assert txt1 == txt2


def test_install_aborta_con_core_hooks_path(tmp_path, monkeypatch):
    monkeypatch.setattr(install_mod, "_core_hooks_path", lambda: "custom/hooks")
    try:
        install_mod.install(hooks_dir=tmp_path, force=False)
    except SystemExit as e:
        assert "core.hooksPath" in str(e)
        return
    raise AssertionError("No aborto con core.hooksPath configurado")


def test_install_force_ignora_core_hooks_path(tmp_path, monkeypatch):
    monkeypatch.setattr(install_mod, "_core_hooks_path", lambda: "custom/hooks")
    acciones = install_mod.install(hooks_dir=tmp_path, force=True)
    assert len(acciones) == 2
    assert (tmp_path / "pre-commit").exists()


# ---- pre_commit_check ----


def _fake_run_factory(returncodes):
    """Devuelve una funcion que consume returncodes en orden."""
    seq = list(returncodes)
    class FakeCompleted:
        def __init__(self, rc):
            self.returncode = rc
            self.stdout = ""
            self.stderr = ""
    def fake(cmd, cwd=None, capture_output=False, text=False, timeout=None):
        rc = seq.pop(0) if seq else 0
        return FakeCompleted(rc)
    return fake


def test_pre_commit_ok(monkeypatch):
    # 6 dirs x 2 checks (pyflakes + compileall) = 12 calls, todos 0.
    monkeypatch.setattr(
        pre_commit_mod.subprocess, "run", _fake_run_factory([0] * 12),
    )
    assert pre_commit_mod.main() == 0


def test_pre_commit_fail_pyflakes(monkeypatch):
    # pyflakes falla en la primera llamada.
    monkeypatch.setattr(
        pre_commit_mod.subprocess, "run", _fake_run_factory([1]),
    )
    assert pre_commit_mod.main() == 1


def test_pre_commit_fail_compileall(monkeypatch):
    # pyflakes OK en todos los dirs, compileall falla en el primero.
    seq = [0] * 6 + [1]
    monkeypatch.setattr(
        pre_commit_mod.subprocess, "run", _fake_run_factory(seq),
    )
    assert pre_commit_mod.main() == 1


# ---- pre_push_check ----


def test_pre_push_skip_env(monkeypatch):
    monkeypatch.setenv("SKIP_PRE_PUSH", "1")
    # subprocess.run no deberia invocarse, pero si por bug, fallara.
    class Boom:
        def __init__(self, *a, **kw):
            raise AssertionError("subprocess.run invocado con SKIP_PRE_PUSH=1")
    monkeypatch.setattr(pre_push_mod.subprocess, "run", Boom)
    assert pre_push_mod.main() == 0


def test_pre_push_ok(monkeypatch):
    monkeypatch.delenv("SKIP_PRE_PUSH", raising=False)
    monkeypatch.setattr(pre_push_mod, "_working_tree_status", lambda: "")
    monkeypatch.setattr(pre_push_mod, "_tiene_xdist", lambda: True)
    class FakeCompleted:
        returncode = 0
    monkeypatch.setattr(
        pre_push_mod.subprocess, "run", lambda *a, **kw: FakeCompleted(),
    )
    assert pre_push_mod.main() == 0


def test_pre_push_fail(monkeypatch):
    monkeypatch.delenv("SKIP_PRE_PUSH", raising=False)
    monkeypatch.setattr(pre_push_mod, "_working_tree_status", lambda: "")
    monkeypatch.setattr(pre_push_mod, "_tiene_xdist", lambda: True)
    class FakeCompleted:
        returncode = 1
    monkeypatch.setattr(
        pre_push_mod.subprocess, "run", lambda *a, **kw: FakeCompleted(),
    )
    assert pre_push_mod.main() == 1


def test_pre_push_timeout(monkeypatch):
    monkeypatch.delenv("SKIP_PRE_PUSH", raising=False)
    monkeypatch.delenv("PRE_PUSH_TIMEOUT", raising=False)
    monkeypatch.setattr(pre_push_mod, "_working_tree_status", lambda: "")
    monkeypatch.setattr(pre_push_mod, "_tiene_xdist", lambda: False)
    def fake_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=600)
    monkeypatch.setattr(pre_push_mod.subprocess, "run", fake_timeout)
    assert pre_push_mod.main() == 1
