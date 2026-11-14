# -*- coding: utf-8 -*-
"""Pre-push hook: suite completa de tests.

Lento (~150s con xdist). Falla si algun test falla.
Se invoca desde .git/hooks/pre-push (stub sh) generado por
install.py. Bypass: git push --no-verify, o env SKIP_PRE_PUSH=1.

xdist se detecta via importlib. Si falta, corre serial con aviso.
Timeout: 600s por defecto, override con PRE_PUSH_TIMEOUT.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_TIMEOUT = 600


def _skip_pedido():
    v = os.environ.get("SKIP_PRE_PUSH", "").strip().lower()
    return v in ("1", "true", "yes")


def _tiene_xdist():
    return importlib.util.find_spec("xdist") is not None


def _working_tree_status():
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    return r.stdout.strip()


def main():
    if _skip_pedido():
        print("[pre-push] SKIP_PRE_PUSH=1, hook bypasseado manualmente",
              file=sys.stderr)
        return 0
    dirty = _working_tree_status()
    if dirty:
        print("[pre-push] AVISO: working tree con cambios sin commit:",
              file=sys.stderr)
        for ln in dirty.splitlines()[:10]:
            print("           " + ln, file=sys.stderr)
        print("[pre-push] El hook corre igualmente", file=sys.stderr)
    timeout = int(os.environ.get("PRE_PUSH_TIMEOUT", DEFAULT_TIMEOUT))
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line", "-m", "not integration_real_data"]
    if _tiene_xdist():
        cmd.extend(["-n", "auto", "--dist", "loadfile"])
    else:
        print("[pre-push] AVISO: xdist no disponible, ejecutando serial",
              file=sys.stderr)
    print("[pre-push] $ " + " ".join(cmd), file=sys.stderr)
    print("[pre-push] timeout=" + str(timeout) + "s", file=sys.stderr)
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout)
    except subprocess.TimeoutExpired:
        print("[pre-push] FAIL timeout " + str(timeout) + "s", file=sys.stderr)
        return 1
    dt = time.time() - t0
    if rc.returncode != 0:
        print("[pre-push] FAIL pytest (%.1fs)" % dt, file=sys.stderr)
        return 1
    print("[pre-push] OK (%.1fs)" % dt, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
