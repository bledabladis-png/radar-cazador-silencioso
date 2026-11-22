# -*- coding: utf-8 -*-
"""FIX 13.11 (2026-11-21, S39): denominadores dinamicos en Breadth.

Antes: b20/b50/b200 se calculaban como X/EXPECTED_SECTOR_COUNT (11) sin
distinguir sectores con datos. b20 quedaba subestimado cuando faltaba
algun sector (sector nuevo, fallo de proveedor).

Ahora: compute_breadth devuelve BreadthResult con denominadores dinamicos
(Series por fecha). El render publica X/Y con Y = sectores con datos, y
declara "X de Y sectores con datos" cuando Y < expected.

Contrato:
- BreadthResult.expected == len(MARKET_TICKERS["sectors"]) (centinela).
- BreadthResult.denom_* = Series con conteo de sectores validos por fecha.
- render: X/Y en cada fila, header con expected y, si aplica, cobertura.
- Legacy artifacts sin denominators_*: fallback a EXPECTED_SECTOR_COUNT.
"""
import numpy as np
import pandas as pd

from config.settings import EXPECTED_SECTOR_COUNT
from config.tickers import MARKET_TICKERS
from indicators.breadth import compute_breadth, BreadthResult
from src.report.breadth import render_breadth_market


# --- Centinela anti-desincronizacion ---

def test_expected_sector_count_matches_market_tickers():
    """EXPECTED_SECTOR_COUNT debe coincidir con len(MARKET_TICKERS["sectors"]).

    Si algun dia se anade un sector a MARKET_TICKERS, este test falla
    y fuerza la actualizacion de config/settings.py. Previene la
    desincronizacion silenciosa entre universo real y constante de config.
    """
    expected_real = len(MARKET_TICKERS["sectors"])
    assert EXPECTED_SECTOR_COUNT == expected_real, (
        f"EXPECTED_SECTOR_COUNT ({EXPECTED_SECTOR_COUNT}) desincronizado de "
        f"MARKET_TICKERS['sectors'] ({expected_real}). "
        "Actualizar config/settings.py al anadir/quitar sector."
    )


# --- compute_breadth: denominadores dinamicos ---

def _df_full(n=300, seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {}
    for s in MARKET_TICKERS["sectors"]:
        close = 100 * np.cumprod(1 + rng.normal(0.0001, 0.015, n))
        data[("Close", s)] = close
        data[("High", s)] = close * 1.01
        data[("Low", s)] = close * 0.99
        data[("Volume", s)] = rng.randint(1_000_000, 5_000_000, n)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_compute_breadth_returns_breadth_result():
    df = _df_full()
    br = compute_breadth(df)
    assert isinstance(br, BreadthResult)
    assert br.expected == len(MARKET_TICKERS["sectors"])


def test_compute_breadth_denoms_full_coverage():
    df = _df_full()
    br = compute_breadth(df)
    expected = len(MARKET_TICKERS["sectors"])
    assert int(br.denom_ema20.iloc[-1]) == expected
    assert int(br.denom_ema50.iloc[-1]) == expected
    assert int(br.denom_ema200.iloc[-1]) == expected
    assert int(br.denom_nhnl.iloc[-1]) == expected


def test_compute_breadth_denoms_partial_coverage():
    df_full = _df_full()
    sectors = MARKET_TICKERS["sectors"]
    keep = set(sectors[:3])
    cols = [c for c in df_full.columns if c[1] in keep]
    df_partial = df_full.loc[:, cols]
    br = compute_breadth(df_partial)
    expected = len(sectors)
    assert br.expected == expected
    assert int(br.denom_ema20.iloc[-1]) == 3
    assert int(br.denom_ema50.iloc[-1]) == 3
    assert int(br.denom_ema200.iloc[-1]) == 3
    assert br.b20.dropna().between(0, 1).all()
    assert br.b200.dropna().between(0, 1).all()


# --- render: header dinamico ---

def _bv_full():
    expected = len(MARKET_TICKERS["sectors"])
    return {
        "% sobre EMA20": 0.545, "% sobre EMA50": 0.454, "% sobre EMA200": 0.363,
        "New Highs (%)": 0.09, "New Lows (%)": 0.18,
        "EMA20 count": 6, "EMA50 count": 5, "EMA200 count": 4,
        "New Highs count": 1, "New Lows count": 2,
        "denominators_ema20": expected,
        "denominators_ema50": expected,
        "denominators_ema200": expected,
        "denominators_nhnl": expected,
        "expected_sector_count": expected,
    }


def test_render_full_coverage_header():
    out = "".join(render_breadth_market(_bv_full()))
    expected = len(MARKET_TICKERS["sectors"])
    assert f"Breadth de Mercado ({expected} sectores)" in out
    assert "con datos" not in out
    assert "Cobertura parcial" not in out


def test_render_partial_coverage_header():
    expected = len(MARKET_TICKERS["sectors"])
    denom = expected - 1
    bv = _bv_full()
    bv["denominators_ema200"] = denom
    bv["denominators_nhnl"] = denom
    out = "".join(render_breadth_market(bv))
    assert f"{denom} de {expected} sectores con datos" in out
    assert "Cobertura parcial" in out


def test_render_legacy_dict_sin_denoms_fallback():
    """Artifact antiguo sin denominators_*: fallback a EXPECTED_SECTOR_COUNT."""
    bv_legacy = {
        "% sobre EMA20": 0.5, "% sobre EMA50": 0.6, "% sobre EMA200": 0.7,
        "New Highs (%)": 0.0, "New Lows (%)": 0.0,
        "EMA20 count": 6, "EMA50 count": 7, "EMA200 count": 8,
        "New Highs count": 0, "New Lows count": 0,
    }
    out = "".join(render_breadth_market(bv_legacy))
    expected = EXPECTED_SECTOR_COUNT
    assert f"Breadth de Mercado ({expected} sectores)" in out
    assert f"6/{expected}" in out


def test_render_partial_uses_denom_in_counts():
    """Cuando cobertura parcial, X/Y usa Y del denominador real."""
    bv = _bv_full()
    bv["denominators_ema200"] = 8
    bv["EMA200 count"] = 4
    out = "".join(render_breadth_market(bv))
    assert "4/8" in out
    assert "4/11" not in out
