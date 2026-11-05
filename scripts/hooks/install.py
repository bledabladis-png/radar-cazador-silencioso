# -*- coding: utf-8 -*-
"""Instala los hooks git (pre-commit, pre-push) en este repo.

Uso: py scripts/hooks/install.py

Escribe stubs sh en .git/hooks/ (o core.hooksPath si esta configurado)
que delegan a pre_commit_check.py / pre_push_check.py. Los stubs
detectan el interprete (py, python3, python) y lo embeben.

Backup .orig si ya existe un hook no-sample. Idempotente.
Aborta si core.hooksPath esta configurado (no pisa config sin --force).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
STUBS = {
    "pre-commit": "pre_commit_check.py",
    "pre-push": "pre_push_check.py",
}


STUB_TEMPLATE = """#!/bin/sh
# Hook generado por scripts/hooks/install.py. No editar a mano.
SCRIPT_DIR="$(git rev-parse --show-toplevel)/scripts/hooks"
if command -v py >/dev/null 2>&1; then
    exec py "$SCRIPT_DIR/{check}" "$@"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 "$SCRIPT_DIR/{check}" "$@"
else
    exec python "$SCRIPT_DIR/{check}" "$@"
fi
"""


def _hooks_dir():
    r = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError("git rev-parse fallo: " + r.stderr.strip())
    p = pathlib.Path(r.stdout.strip())
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _core_hooks_path():
    r = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    v = r.stdout.strip()
    return v or None


def _backup_si_existe(p):
    if not p.exists():
        return None
    if p.name.endswith(".sample"):
        return None
    bak = pathlib.Path(str(p) + ".orig")
    shutil.copy2(p, bak)
    return bak


def install(hooks_dir=None, force=False):
    """Instala los 2 stubs. Devuelve lista de (path, accion)."""
    if hooks_dir is None:
        hooks_dir = _hooks_dir()
    hooks_dir = pathlib.Path(hooks_dir)
    hooks_dir.mkdir(parents=True, exist_ok=True)
    chp = _core_hooks_path()
    if chp and not force:
        raise SystemExit(
            "core.hooksPath esta configurado (" + chp + "). "
            "Instala alli o usa --force para ignorar."
        )
    acciones = []
    for name, check in STUBS.items():
        dst = hooks_dir / name
        bak = _backup_si_existe(dst)
        contenido = STUB_TEMPLATE.format(check=check)
        dst.write_text(contenido, encoding="utf-8", newline="\n")
        try:
            os.chmod(dst, 0o755)
        except OSError:
            pass
        accion = "instalado"
        if bak:
            accion += " (backup: " + bak.name + ")"
        acciones.append((str(dst), accion))
    return acciones


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="ignora core.hooksPath configurado")
    args = ap.parse_args(argv)
    for path, accion in install(force=args.force):
        print(path + " -> " + accion)
    print("OK. Bypass: git commit --no-verify / git push --no-verify / SKIP_PRE_PUSH=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
