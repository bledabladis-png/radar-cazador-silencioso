# -*- coding: utf-8 -*-
"""Tests muestreo B v8 (4 celdas, SRSWOR, Hamilton, censos).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.sampling_b_v8 import (
    asignar_celdas,
    asignar_n_h,
    build_estratos_v8,
    sample_b_v8,
    totales_por_celda,
    verificar_factibilidad_minimos,
    _hamilton,
)


def _make_meta(n=400, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    return pd.DataFrame({
        "ticker": rng.choice(["A", "B", "C", "D"], size=n),
        "t": idx,
        "sector": rng.choice(["XLK", "XLF"], size=n),
        "periodo": rng.choice(["2021-2022", "2023-2026"], size=n),
        "detect_sow": rng.integers(0, 2, size=n),
        "ctx": rng.integers(0, 2, size=n).astype(bool),
    })


def test_asignar_celdas_correcto():
    df = pd.DataFrame({
        "detect_sow": [1, 1, 0, 0],
        "ctx": [True, False, True, False],
    })
    out = asignar_celdas(df)
    assert list(out["celda"]) == ["C1", "C2", "C3", "C4"]


def test_asignar_celdas_falta_columna():
    df = pd.DataFrame({"detect_sow": [1]})
    with pytest.raises(ValueError, match="Falta columna"):
        asignar_celdas(df)


def test_build_estratos_v8_cuenta():
    m = _make_meta(200)
    m = asignar_celdas(m)
    e = build_estratos_v8(m)
    assert "celda" in e.columns
    assert e["N_h"].sum() == len(m)
    assert set(e["celda"].unique()).issubset({"C1", "C2", "C3", "C4"})


def test_hamilton_suma_exacta():
    cuotas = np.array([1.3, 2.7, 0.5, 4.5])
    total = 9
    res = _hamilton(cuotas, total)
    assert res.sum() == total


def test_hamilton_caso_simple():
    cuotas = np.array([10.0, 5.0, 5.0])
    res = _hamilton(cuotas, 20)
    assert res.sum() == 20
    assert res[0] >= res[1]


def test_verificar_factibilidad_ok():
    e = pd.DataFrame({
        "celda": ["C1"] * 3,
        "sector": ["XLK", "XLF", "XLU"],
        "periodo": ["P1", "P1", "P1"],
        "N_h": [100, 100, 100],
    })
    r = verificar_factibilidad_minimos(e, K=200)
    assert r["ok"] is True


def test_verificar_factibilidad_falla():
    # 50 estratos con N_h=100 -> minimos 5*50 = 250 > 200
    e = pd.DataFrame({
        "celda": ["C1"] * 50,
        "sector": [f"S{i}" for i in range(50)],
        "periodo": ["P1"] * 50,
        "N_h": [100] * 50,
    })
    r = verificar_factibilidad_minimos(e, K=200)
    assert r["ok"] is False
    assert r["celda"] == "C1"

def test_asignar_n_h_respeta_K_por_celda():
    e = pd.DataFrame({
        "celda": ["C1"] * 4 + ["C2"] * 4,
        "sector": ["XLK", "XLF", "XLU", "XLE"] * 2,
        "periodo": ["P1"] * 8,
        "N_h": [100, 200, 150, 120] * 2,
    })
    out = asignar_n_h(e, K=200)
    for celda in ("C1", "C2"):
        sub = out[out["celda"] == celda]
        assert sub["n_h"].sum() == 200


def test_asignar_n_h_minimo_cinco():
    e = pd.DataFrame({
        "celda": ["C1"] * 4,
        "sector": ["A", "B", "C", "D"],
        "periodo": ["P1"] * 4,
        "N_h": [50, 50, 50, 50],
    })
    out = asignar_n_h(e, K=200)
    # 4 estratos con N_h=50; minimo 5 cada uno; suma 20; remanente 180 repartido
    assert (out["n_h"] >= 5).all()
    assert out["n_h"].sum() == 200


def test_asignar_n_h_censo_estrato_pequeno():
    e = pd.DataFrame({
        "celda": ["C1"] * 3,
        "sector": ["A", "B", "C"],
        "periodo": ["P1"] * 3,
        "N_h": [100, 3, 100],  # B tiene 3 -> censo
    })
    out = asignar_n_h(e, K=200)
    row_b = out[out["sector"] == "B"].iloc[0]
    assert row_b["n_h"] == 3  # censo, no bootstrap


def test_asignar_n_h_nunca_excede_N_h():
    e = pd.DataFrame({
        "celda": ["C1"] * 2,
        "sector": ["A", "B"],
        "periodo": ["P1"] * 2,
        "N_h": [10, 500],
    })
    out = asignar_n_h(e, K=200)
    assert (out["n_h"] <= out["N_h"]).all()


def test_sample_b_v8_produce_n_total():
    m = _make_meta(400)
    m = asignar_celdas(m)
    e = build_estratos_v8(m)
    e = asignar_n_h(e, K=50)  # K bajo para test rapido
    out = sample_b_v8(m, e, seed=42)
    assert len(out) > 0
    assert "w_h" in out.columns
    assert "N_h" in out.columns
    assert "n_h" in out.columns


def test_sample_b_v8_pesos_conocidos():
    m = _make_meta(400)
    m = asignar_celdas(m)
    e = build_estratos_v8(m)
    e = asignar_n_h(e, K=50)
    out = sample_b_v8(m, e, seed=42)
    for _, row in out.iterrows():
        assert abs(row["w_h"] - row["N_h"] / row["n_h"]) < 1e-9


def test_sample_b_v8_determinista():
    m = _make_meta(400)
    m = asignar_celdas(m)
    e = build_estratos_v8(m)
    e = asignar_n_h(e, K=50)
    o1 = sample_b_v8(m, e, seed=7)
    o2 = sample_b_v8(m, e, seed=7)
    pd.testing.assert_frame_equal(o1, o2)


def test_sample_b_v8_sin_duplicados():
    m = _make_meta(400)
    m = asignar_celdas(m)
    e = build_estratos_v8(m)
    e = asignar_n_h(e, K=50)
    out = sample_b_v8(m, e, seed=42)
    # Sin reemplazo dentro de cada estrato: no debe haber duplicados (ticker, t)
    dup = out.duplicated(subset=["ticker", "t"]).any()
    assert not dup


def test_totales_por_celda():
    m = _make_meta(400)
    m = asignar_celdas(m)
    e = build_estratos_v8(m)
    e = asignar_n_h(e, K=50)
    t = totales_por_celda(e)
    for celda in ("C1", "C2", "C3", "C4"):
        assert celda in t
        assert t[celda]["n_h_total"] == 50