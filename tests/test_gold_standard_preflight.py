# -*- coding: utf-8 -*-
"""Tests preflight integration (P1.6, dictamen).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.preflight import run_preflight

PARQUET = ROOT / "data" / "stock_prices.parquet"


@pytest.mark.skipif(
    not PARQUET.exists(),
    reason="stock_prices.parquet no presente",
)
def test_preflight_completo_sobre_datos_reales():
    """Ejecuta la cadena completa sobre datos reales."""
    report = run_preflight(n_B_test=100, verbose=False)
    assert "checks_ok" in report
    assert report["checks_ok"]["frame_no_filtra_sector"] is True
    assert report["checks_ok"]["A_B_disjuntas"] is True
    assert report["checks_ok"]["blind_ids_unicos"] is True


@pytest.mark.skipif(
    not PARQUET.exists(),
    reason="stock_prices.parquet no presente",
)
def test_preflight_reporta_estructura():
    report = run_preflight(n_B_test=100, verbose=False)
    for clave in (
        "dataset", "sector_map", "sampling_frame", "metadata",
        "estratos", "muestra_B", "muestra_A", "disjuncion", "blind_ids",
    ):
        assert clave in report["pasos"]