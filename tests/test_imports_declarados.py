# -*- coding: utf-8 -*-
"""Valida que todo import top-level no-stdlib ni local esta declarado.

Regresion 2026-10-16/17: el commit 529f4ed2 introdujo `import yaml`
sin declarar pyyaml en requirements.txt. En local pasaba porque optuna
lo arrastraba como dependencia transitiva. En CI los 3 modulos que
importan yaml fallaban en collection (ModuleNotFoundError) y el run
diario murio en "Run tests".

Este test barre todos los .py de src/scripts/tests/validation, recoge
imports top-level, y falla si alguno no es stdlib, ni local, ni esta
en requirements.txt. Coste ~0.5s.
"""
import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

IMPORT_TO_PIP = {
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "Crypto": "pycryptodome",
    "websocket": "websocket-client",
    "pandas_datareader": "pandas-datareader",
}

SCAN_DIRS = ("src", "scripts", "tests", "validation")


def _read_requirements():
    declared = set()
    path = ROOT / "requirements.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if m:
            declared.add(m.group(1).lower().replace("_", "-"))
    return declared


def _local_names():
    """Top-level names resolubles localmente.

    Incluye directorios raiz con __init__.py (data, indicators,
    regimes, config, scripts, tests, validation, src) y sus
    subpaquetes, y ficheros .py sueltos en raiz.
    """
    names = set()
    for p in ROOT.iterdir():
        if p.name.startswith(".") or p.name == "__pycache__":
            continue
        if p.is_dir() and (p / "__init__.py").exists():
            names.add(p.name)
        elif p.suffix == ".py" and p.stem != "__init__":
            names.add(p.stem)
    src = ROOT / "src"
    if src.is_dir():
        for p in src.iterdir():
            if p.is_dir() and (p / "__init__.py").exists():
                names.add(p.name)
            elif p.suffix == ".py" and p.stem != "__init__":
                names.add(p.stem)
    return names


def _iter_py_files():
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in base.rglob("*.py"):
            yield p


def _top_level_imports(path):
    try:
        tree = ast.parse(path.read_bytes())
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                yield n.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]


def test_todos_los_imports_declarados():
    stdlib = set(sys.stdlib_module_names)
    locals_ = _local_names()
    declared = _read_requirements()
    missing = {}
    for p in _iter_py_files():
        for name in _top_level_imports(p):
            if not name or name.startswith("_"):
                continue
            if name in stdlib or name in locals_:
                continue
            pip_name = IMPORT_TO_PIP.get(name, name).lower().replace("_", "-")
            if pip_name not in declared:
                rel = p.relative_to(ROOT).as_posix()
                missing.setdefault((name, pip_name), set()).add(rel)
    if missing:
        lines = []
        for (imp, pip), files in sorted(missing.items()):
            examples = sorted(files)[:3]
            lines.append(
                "  import {0!r} -> pip {1!r} no en requirements.txt. "
                "Ej: {2}".format(imp, pip, ", ".join(examples))
            )
        pytest.fail(
            "Imports no declarados en requirements.txt:\n"
            + "\n".join(lines)
        )


def test_detecta_import_faltante_con_mock(monkeypatch):
    """Negativo: sin requirements y sin mappings, el test falla.

    Demuestra que el positivo caza imports reales y no pasa en vacio.
    """
    monkeypatch.setattr(
        "tests.test_imports_declarados._read_requirements",
        lambda: set(),
    )
    monkeypatch.setattr(
        "tests.test_imports_declarados.IMPORT_TO_PIP",
        {},
    )
    with pytest.raises(pytest.fail.Exception) as exc_info:
        test_todos_los_imports_declarados()
    msg = str(exc_info.value)
    assert any(
        name in msg for name in ("numpy", "pandas", "yaml", "requests",
                                 "yfinance", "pytest", "bs4", "openpyxl")
    ), "el negativo debe mencionar imports reales: {0}".format(msg[:300])