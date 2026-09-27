"""Tests del CftcProvider (F5.7-14). Sin red."""
from __future__ import annotations

import ast
from pathlib import Path


def test_cftc_data_importa_thresholds_desde_settings():
    """AST: cftc_data.py debe leer los thresholds desde config.settings."""
    p = Path(__file__).resolve().parent.parent / "data" / "providers" / "cftc_data.py"
    tree = ast.parse(p.read_text(encoding="utf-8-sig"))
    importados = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "config.settings":
            for alias in node.names:
                importados.add(alias.name)
    assert "CFTC_HISTORY_DAYS" in importados
    assert "CFTC_ACTIVE_CONTRACT_DAYS" in importados


def test_cftc_thresholds_en_settings():
    from config.settings import CFTC_HISTORY_DAYS, CFTC_ACTIVE_CONTRACT_DAYS
    assert CFTC_HISTORY_DAYS == 365
    assert CFTC_ACTIVE_CONTRACT_DAYS == 30


def test_cftc_data_sin_hardcodes_365_30():
    """AST: no deben quedar Timedelta(days=365) ni Timedelta(days=30) literales."""
    p = Path(__file__).resolve().parent.parent / "data" / "providers" / "cftc_data.py"
    src = p.read_text(encoding="utf-8-sig")
    assert "Timedelta(days=365)" not in src
    assert "Timedelta(days=30)" not in src
