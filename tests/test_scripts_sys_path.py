# -*- coding: utf-8 -*-
"""Verifica que los scripts con `import src.*` funcionan invocados desde
cwd=scripts, reproduciendo el fallo de CI.

Contexto (2026-10-01): los workflows invocan `python scripts/X.py`, lo
que pone `scripts/` en sys.path[0] pero NO la raiz del repo. Los scripts
que hacen `from src.holdings_filter import ...` necesitan insertar ROOT
en sys.path. Este bug cego 3 workflows trimestrales: primer schedule
real, los 3 con ModuleNotFoundError.

Verificacion: subprocess con cwd=scripts, runpy con run_name !=
'__main__' para no disparar main() (que hace GETs reales).
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

SCRIPTS_CON_SRC = [
    "update_sector_holdings.py",
    "update_index_holdings.py",
    "parse_ssga_fez.py",
    "parse_blackrock_final.py",
]


def test_script_importa_src_desde_cwd_scripts():
    """Con cwd=scripts, sys.path[0]=cwd apunta a scripts/. `src/` no esta
    en sys.path. El sys.path.insert del propio script debe hacer que el
    import funcione sin ejecutar main().
    """
    fallos = []
    for name in SCRIPTS_CON_SRC:
        code = (
            "import runpy; "
            "runpy.run_path({0!r}, run_name='__test_import__')"
        ).format(name)
        r = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(SCRIPTS_DIR),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            fallos.append((name, r.returncode, r.stderr))
    assert not fallos, (
        "Scripts que fallan al importar src.* desde cwd=scripts:\n"
        + "\n".join(
            "{0}: exit={1}\nstderr:\n{2}".format(n, rc, err)
            for n, rc, err in fallos
        )
    )
