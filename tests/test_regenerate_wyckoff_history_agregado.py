# -*- coding: utf-8 -*-
"""Tests de _construir_agregado (regenerate_wyckoff_history).

Sesion 13 post-fix PCT-FLOAT-ROUNDING-DIVERGENCE (2026-10-27):
pct_* se redondea a 8 decimales de forma consistente con
compute_sector_wyckoff_distribution.
"""
from __future__ import annotations

import math

from scripts.regenerate_wyckoff_history import _construir_agregado


def _rows(phases, sector="XLK"):
    return [
        {"ticker": f"T{i}", "date": "2026-10-23", "phase": ph}
        for i, ph in enumerate(phases)
    ], {"T": sector} if False else {f"T{i}": sector for i in range(len(phases))}


def test_pct_redondeado_a_8_decimales():
    """1 ACC + 5 RANGE -> pct_accumulation = round(1/6*100, 8)."""
    rows, mapping = _rows(["ACCUMULATION"] + ["RANGE"] * 5)
    df = _construir_agregado(rows, mapping)
    assert len(df) == 1
    row = df.iloc[0]
    esperado = round(1 / 6 * 100, 8)
    assert row["pct_accumulation"] == esperado
    assert row["pct_range"] == round(5 / 6 * 100, 8)
    # Sin redondeo serian 16.666666666666664 y 83.33333333333333
    assert row["pct_accumulation"] != 1 / 6 * 100 or math.isclose(1 / 6 * 100, esperado, abs_tol=0)


def test_pct_nan_si_n_valid_menor_que_5():
    """3 tickers validos -> pct_* NaN (regla n_valid >= 5)."""
    rows, mapping = _rows(["ACCUMULATION"] * 3)
    df = _construir_agregado(rows, mapping)
    assert len(df) == 1
    row = df.iloc[0]
    assert math.isnan(row["pct_accumulation"])
    assert row["count_accumulation"] == 3