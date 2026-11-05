# -*- coding: utf-8 -*-
"""Pre-commit hook: pyflakes + compileall sobre codigo productivo.

Rapido (~10s). Falla si pyflakes reporta algo o compileall falla.
Se invoca desde .git/hooks/pre-commit (stub sh) generado por
install.py. Bypass: git commit --no-verify.

Roots: solo codigo productivo. outputs/ y data/ excluidos
(artefactos de proveedores con calendarios no-NYSE).
"""
from __future__ import annotations

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CODE_DIRS = ("src", "scripts", "tests", "config", "regimes", "indicators")


def _run(cmd):
    print("[pre-commit] $ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, cwd=str(ROOT))


def main():
    for d in CODE_DIRS:
        p = ROOT / d
        if not p.exists():
            continue
        rc = _run([sys.executable, "-m", "pyflakes", d])
        if rc.returncode != 0:
            print("[pre-commit] FAIL pyflakes " + d, file=sys.stderr)
            return 1
    for d in CODE_DIRS:
        p = ROOT / d
        if not p.exists():
            continue
        rc = _run([sys.executable, "-m", "compileall", "-q", d])
        if rc.returncode != 0:
            print("[pre-commit] FAIL compileall " + d, file=sys.stderr)
            return 1
    print("[pre-commit] OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
