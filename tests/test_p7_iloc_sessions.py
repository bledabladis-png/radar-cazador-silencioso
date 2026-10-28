"""Regresion P7-ROLLING-WINDOW-GAPS en cross_asset_context y options.

Fix 2026-10-28: reemplazar `iloc[-N:]` por slice_last_n_sessions.
Test AST acotado: verifica que los ficheros no vuelven al patron.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _source(rel_path):
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_cross_asset_context_no_usa_iloc_negativo():
    src = _source("indicators/cross_asset_context.py")
    hits = re.findall(r"\biloc\[-\w+:\]", src)
    assert hits == [], (
        "cross_asset_context.py vuelve a usar iloc[-N:] sobre panel con "
        "gaps. Usar slice_last_n_sessions. Encontrados: " + str(hits)
    )


def test_options_no_usa_iloc_negativo_sobre_pcr():
    """Permitir iloc[-1] (valor unico, no ventana). Rechazar
    iloc[-N:] con N > 1 sobre pcr_ewm.
    """
    src = _source("indicators/options.py")
    # iloc[-1] esta permitido. Buscar iloc[-N:] con N>1.
    hits = re.findall(r"\.iloc\[-(?!1\])[\w]+:\]", src)
    assert hits == [], (
        "options.py vuelve a usar iloc[-N:] sobre pcr_ewm. Usar "
        "slice_last_n_sessions. Encontrados: " + str(hits)
    )


def test_cross_asset_context_importa_helper():
    src = _source("indicators/cross_asset_context.py")
    assert "slice_last_n_sessions" in src, (
        "cross_asset_context.py no importa slice_last_n_sessions"
    )


def test_options_importa_helper():
    src = _source("indicators/options.py")
    assert "slice_last_n_sessions" in src, (
        "options.py no importa slice_last_n_sessions"
    )
