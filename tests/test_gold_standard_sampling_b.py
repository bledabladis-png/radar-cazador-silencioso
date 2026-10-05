# -*- coding: utf-8 -*-
"""Tests muestreo B.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.sampling_b import (
    build_estratos,
    collapse_estratos,
    proportional_allocation,
    sample_b,
)


def _make_frame(n_per_cell=50, sectors=("Tech", "Health", "Energy"),
                periodos=("2021-2022", "2023-2024", "2025-2026")):
    rows = []
    base = pd.Timestamp("2021-01-04")
    for i, s in enumerate(sectors):
        for j, p in enumerate(periodos):
            for k in range(n_per_cell):
                rows.append({
                    "ticker": f"{s[:2]}{j}",
                    "t": base + pd.Timedelta(days=k + j * 400),
                    "sector": s,
                    "periodo": p,
                })
    return pd.DataFrame(rows)


def test_build_estratos_cuenta_celdas():
    frame = _make_frame(n_per_cell=10)
    e = build_estratos(frame)
    assert len(e) == 9
    assert e["N_h"].sum() == 90


def test_collapse_estratos_sin_colapso_cuando_suficiente():
    frame = _make_frame(n_per_cell=10)
    e = build_estratos(frame)
    out = collapse_estratos(e, n_min=5)
    assert out["N_h"].min() >= 5


def test_collapse_estratos_activo_cuando_pequeno():
    """Dejar 1 sola fila en una celda y verificar que se colapsa."""
    frame = _make_frame(n_per_cell=50, sectors=("Tech", "Health", "Energy"))
    mask = (frame["sector"] == "Energy") & (frame["periodo"] == "2025-2026")
    cell = frame[mask]
    frame_small = pd.concat([frame[~mask], cell.head(1)], axis=0)
    e = build_estratos(frame_small)
    assert (e["N_h"] < 5).any()
    out = collapse_estratos(e, n_min=5)
    assert out["N_h"].min() >= 5


def test_proportional_allocation_suma_n():
    frame = _make_frame(n_per_cell=50)
    e = build_estratos(frame)
    out = proportional_allocation(e, n_total=200)
    assert out["n_h"].sum() == 200
    assert (out["n_h"] <= out["N_h"]).all()


def test_proportional_allocation_respeta_rao_wu():
    frame = _make_frame(n_per_cell=5)
    e = build_estratos(frame)
    out = proportional_allocation(e, n_total=80)
    assert (out["n_h"] >= 2).all()


def test_sample_b_devuelve_tamano_correcto():
    frame = _make_frame(n_per_cell=100)
    e = build_estratos(frame)
    out = sample_b(frame, n_B=200, estratos_finales=e, seed=1)
    assert len(out) == 200
    assert (out["muestra"] == "B").all()


def test_sample_b_pesos_conocidos():
    frame = _make_frame(n_per_cell=100)
    e = build_estratos(frame)
    out = sample_b(frame, n_B=180, estratos_finales=e, seed=1)
    for _, row in out.iterrows():
        expected = row["N_h"] / row["n_h"]
        assert abs(row["w_h"] - expected) < 1e-9


def test_sample_b_determinista():
    frame = _make_frame(n_per_cell=100)
    e = build_estratos(frame)
    out1 = sample_b(frame, n_B=200, estratos_finales=e, seed=3)
    out2 = sample_b(frame, n_B=200, estratos_finales=e, seed=3)
    pd.testing.assert_frame_equal(out1, out2)


def test_sample_b_no_condicionado_por_detector():
    frame = _make_frame(n_per_cell=100)
    assert "detect_sow" not in frame.columns
    e = build_estratos(frame)
    out = sample_b(frame, n_B=200, estratos_finales=e, seed=1)
    assert "detect_sow" not in out.columns


def test_sample_b_sin_min_gap():
    frame = _make_frame(n_per_cell=100)
    e = build_estratos(frame)
    out = sample_b(frame, n_B=300, estratos_finales=e, seed=1)
    assert out["ticker"].nunique() < len(out)