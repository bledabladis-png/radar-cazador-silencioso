# -*- coding: utf-8 -*-
"""Tests colapso por periodo sin fusionar sectores (P1.5, dictamen).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.sampling_b import collapse_estratos


def _estratos(sectores_periodos):
    """sectores_periodos: lista de (sector, periodo, N_h)."""
    return pd.DataFrame(
        sectores_periodos, columns=["sector", "periodo", "N_h"]
    )


def test_sin_colapso_si_todas_las_celdas_grandes():
    e = _estratos([
        ("XLK", "2021-2022", 100),
        ("XLK", "2023-2024", 80),
        ("XLF", "2021-2022", 90),
        ("XLF", "2023-2024", 70),
    ])
    out = collapse_estratos(e, n_min=5)
    # Sin cambios
    assert len(out) == 4
    assert set(out["sector"]) == {"XLK", "XLF"}


def test_colapso_periodo_si_celda_pequena():
    e = _estratos([
        ("XLK", "2021-2022", 100),
        ("XLK", "2023-2024", 3),   # pequeña -> colapsa periodos de XLK
        ("XLF", "2021-2022", 80),
        ("XLF", "2023-2024", 70),
    ])
    out = collapse_estratos(e, n_min=5)
    # XLK debe tener una sola fila con periodo="global"
    xlk = out[out["sector"] == "XLK"]
    assert len(xlk) == 1
    assert xlk["periodo"].iloc[0] == "global"
    assert xlk["N_h"].iloc[0] == 103
    # XLF intacto (2 periodos)
    xlf = out[out["sector"] == "XLF"]
    assert len(xlf) == 2


def test_no_fusiona_sectores_distintos():
    e = _estratos([
        ("XLK", "2021-2022", 100),
        ("XLF", "2023-2024", 3),   # XLF pequeño
    ])
    out = collapse_estratos(e, n_min=5)
    # XLF no debe fusionarse con XLK
    sectores = set(out["sector"])
    assert "XLK" in sectores
    # XLF tras colapso sigue con N_h=3 < 5 -> OTHER
    assert "OTHER" in sectores
    assert "XLF" not in sectores


def test_sector_other_si_persiste_pequeno():
    e = _estratos([
        ("XLK", "2021-2022", 100),
        ("ZZZ", "2021-2022", 1),
        ("ZZZ", "2023-2024", 1),
    ])
    out = collapse_estratos(e, n_min=5)
    zzz = out[out["sector"] == "OTHER"]
    assert len(zzz) == 1
    assert zzz["N_h"].iloc[0] == 2


def test_determinista():
    e = _estratos([
        ("XLK", "2021-2022", 100),
        ("XLK", "2023-2024", 3),
        ("XLF", "2021-2022", 80),
    ])
    o1 = collapse_estratos(e, n_min=5)
    o2 = collapse_estratos(e, n_min=5)
    pd.testing.assert_frame_equal(o1, o2)