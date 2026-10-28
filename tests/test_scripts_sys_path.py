"""Test dinamico de sys.path para scripts con `import src.*`.

Contexto (2026-10-01): los workflows invocan `python scripts/X.py`,
lo que pone `scripts/` en sys.path[0] pero NO la raiz del repo. Los
scripts que hacen `from src.X import ...` necesitan insertar ROOT en
sys.path. Este bug cego 3 workflows trimestrales (c4f99795).

Bug 2026-10-28 (TEST-SCRIPTS-SYS-PATH-DYNAMIC): el test anterior
hardcodeaba 5 scripts. Si un sexto entraba sin sys.path, el test no
lo detectaba. Ahora se descubren dinamicamente via AST (27 a fecha
de hoy, excluyendo diag_*).

Estrategia:
- Fuente de verdad: `git ls-files scripts/**/*.py` (solo versionados).
  Los scripts diag_* estan gitignored y no se testean aqui.
- Descubrimiento via AST (no regex, para evitar falsos positivos
  dentro de strings/comentarios). Cubre imports en cualquier nivel
  del arbol (top-level o dentro de try).
- Verificacion via subprocess con cwd=scripts y runpy con
  run_name!='__main__' (no dispara main, que hace GETs reales).
- Deteccion de sys.path: reconoce alias (`import sys as _sys`,
  `from sys import path`) y la forma `if "X" not in sys.path:`.
- Tests:
  * Scanner encuentra los 5 originales (sanity).
  * Todos los encontrados tienen `sys.path` en el codigo.
  * Todos se importan OK desde cwd=scripts.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
ROOT = Path(__file__).resolve().parents[1]

# Los 5 scripts que cubria el test hardcodeado anterior. Si el scanner
# deja de encontrarlos, este test falla con mensaje claro.
SCRIPT_ORIGINALES_HARDCODED = [
    "update_sector_holdings.py",
    "update_index_holdings.py",
    "update_european_holdings.py",
    "parse_ssga_fez.py",
    "parse_blackrock_final.py",
]


def _parse(p):
    """Parsea un .py con BOM-tolerant utf-8-sig. None si SyntaxError."""
    try:
        return ast.parse(p.read_bytes().decode("utf-8-sig"))
    except (OSError, SyntaxError):
        return None


def _imports_src(tree):
    """True si hay import src.* / from src.* en cualquier parte."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "src" or a.name.startswith("src."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "src" or mod.startswith("src."):
                return True
    return False


def _alias_de_sys(tree):
    """Nombres con los que se referencia el modulo sys en el script.
    Incluye 'sys' (import sys), alias (_sys, sys_mod...) y el nombre
    'path' si se hizo from sys import path.
    """
    nombres = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "sys":
                    nombres.add(a.asname or "sys")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") == "sys":
                for a in node.names:
                    if a.name == "path":
                        nombres.add("path-from-sys")
    return nombres


def _tiene_syspath(tree):
    """True si hay sys.path.insert/append en cualquier parte, con sys
    o cualquiera de sus alias.
    """
    alias = _alias_de_sys(tree)
    if not alias:
        return False

    def _es_atributo_path(node, nombre_modulo):
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "path"
            and isinstance(node.value, ast.Name)
            and node.value.id == nombre_modulo
        )

    for node in ast.walk(tree):
        # sys.path.insert(...) o _sys.path.insert(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            f = node.func
            if f.attr in ("insert", "append"):
                for nombre in alias:
                    if nombre == "path-from-sys":
                        continue
                    if _es_atributo_path(f.value, nombre):
                        return True
        # from sys import path; path.insert(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            f = node.func
            if f.attr in ("insert", "append") and isinstance(f.value, ast.Name):
                if f.value.id == "path" and "path-from-sys" in alias:
                    return True
    return False


def _tracked_scripts():
    """Lista de Paths (relativos al repo) de scripts/**/*.py versionados
    en git. Excluye diag_* (gitignored, no repo). git ls-files es la
    fuente de verdad: si no esta en git, no se testea.
    """
    r = subprocess.run(
        ["git", "ls-files", "scripts/"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.endswith(".py"):
            continue
        p = Path(line)
        if p.name.startswith("diag_"):
            continue
        out.append(p)
    return sorted(out)


def _descubrir():
    """Scripts versionados con import src.* en cualquier nivel."""
    out = []
    for p in _tracked_scripts():
        tree = _parse(ROOT / p)
        if tree is None:
            continue
        if _imports_src(tree):
            out.append(p)
    return out


def test_scanner_encuentra_los_5_originales():
    paths = _descubrir()
    nombres = {p.name for p in paths}
    faltan = [n for n in SCRIPT_ORIGINALES_HARDCODED if n not in nombres]
    assert not faltan, (
        f"El scanner no encuentra los scripts originales {faltan}. "
        f"Encontrados: {sorted(nombres)}"
    )


def test_scanner_encuentra_al_menos_20():
    paths = _descubrir()
    assert len(paths) >= 20, (
        f"Solo {len(paths)} scripts con import src.*. "
        f"Revisar scanner o scripts: {[p.name for p in paths]}"
    )


def test_todos_los_scripts_con_src_tienen_syspath():
    paths = _descubrir()
    sin_syspath = []
    for p in paths:
        tree = _parse(ROOT / p)
        if tree is None:
            continue
        if not _tiene_syspath(tree):
            sin_syspath.append(str(p))
    assert not sin_syspath, (
        "Scripts con import src.* sin sys.path.insert/append: "
        + str(sin_syspath)
        + ". Anadir sys.path.insert(0, str(ROOT)) antes del import."
    )


def test_todos_los_scripts_con_src_se_importan_desde_cwd_scripts():
    paths = _descubrir()
    fallos = []
    for p in paths:
        rel = p.relative_to("scripts")
        rel_posix = str(rel).replace("\\", "/")
        code = (
            "import runpy; "
            "runpy.run_path({0!r}, run_name='__test_import__')"
        ).format(rel_posix)
        r = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(SCRIPTS_DIR),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode != 0:
            fallos.append((rel_posix, r.returncode, r.stderr[-500:]))
    assert not fallos, (
        "Scripts que fallan al importar src.* desde cwd=scripts:\n"
        + "\n".join(
            "{0}: exit={1}\nstderr:\n{2}".format(n, rc, err)
            for n, rc, err in fallos
        )
    )