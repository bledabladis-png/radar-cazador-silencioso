# -*- coding: utf-8 -*-
"""Verifica que los 3 scripts de update llamen al snapshot PIT.

PIT-fix (auditoria externa 2026-10-25): update_sector_holdings.py,
update_index_holdings.py y update_european_holdings.py deben llamar a
compute_and_persist_holdings_snapshot tras escribir el CSV.

Test AST: no importa los scripts (que harian descargas). Solo parsea
el codigo fuente y verifica la presencia de la llamada.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    "scripts/update_sector_holdings.py",
    "scripts/update_index_holdings.py",
    "scripts/update_european_holdings.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_llama_snapshot(script):
    """El script debe llamar a compute_and_persist_holdings_snapshot."""
    path = ROOT / script
    if not path.exists():
        pytest.skip(f"{script} no existe")
    src = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(src)

    found_call = False
    found_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "compute_and_persist_holdings_snapshot":
                found_call = True
            elif isinstance(fn, ast.Attribute) and fn.attr == "compute_and_persist_holdings_snapshot":
                found_call = True
        if isinstance(node, ast.ImportFrom):
            if node.module == "src.pipeline.holdings_snapshot":
                for alias in node.names:
                    if alias.name == "compute_and_persist_holdings_snapshot":
                        found_import = True

    assert found_import, f"{script}: no importa compute_and_persist_holdings_snapshot"
    assert found_call, f"{script}: no llama a compute_and_persist_holdings_snapshot"


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_wrap_en_try_except(script):
    """La llamada debe estar protegida por try/except (no romper update)."""
    path = ROOT / script
    if not path.exists():
        pytest.skip(f"{script} no existe")
    src = path.read_text(encoding="utf-8")
    # Heuristica: la llamada esta dentro de un bloque try que captura
    # al menos RuntimeError o ValueError.
    idx = src.find("compute_and_persist_holdings_snapshot(")
    if idx < 0:
        pytest.fail(f"{script}: no contiene llamada")
    # Buscar 'try:' 200 chars antes y 'except' 500 chars despues
    context_before = src[max(0, idx - 300):idx]
    context_after = src[idx:idx + 600]
    assert "try:" in context_before, f"{script}: sin 'try:' antes de la llamada"
    assert "except" in context_after, f"{script}: sin 'except' despues de la llamada"
