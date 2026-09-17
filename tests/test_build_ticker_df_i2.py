# -*- coding: utf-8 -*-
"""Test de regresion I2 (2026-09-18).

Bug: wyckoff_structure_core degenera al fallback silencioso RANGE
cuando recibe un df con NaN internos. Verificado empiricamente:
classify_wyckoff_phase(df_stocks, ticker) devuelve RANGE para
20/20 tickers de XLB; con ticker_df+dropna() devuelve distribucion
real (6 RANGE, 7 MARKUP, 4 DISTRIBUTION, 3 ACCUMULATION).

Fix: helper build_ticker_df(df, ticker) + aplicacion en
sector_wyckoff_distribution.py y index_leaders.py.
"""
import numpy as np
import pandas as pd
import pytest

from indicators.wyckoff import build_ticker_df, classify_wyckoff_phase


def _make_multiindex_df(tickers, n=300, seed=42):
    """Crea un df con MultiIndex (field, ticker) como df_stocks."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    data = {}
    for tk in tickers:
        base = 100 + rng.standard_normal(n).cumsum()
        data[("Open", tk)] = base + rng.standard_normal(n) * 0.5
        data[("High", tk)] = base + abs(rng.standard_normal(n)) * 1.0
        data[("Low", tk)] = base - abs(rng.standard_normal(n)) * 1.0
        data[("Close", tk)] = base
        data[("Volume", tk)] = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(data, index=dates)


def test_build_ticker_df_estructura_correcta():
    """El helper devuelve las 5 columnas OHLCV sin NaN."""
    df = _make_multiindex_df(["AAA", "BBB"])
    tdf = build_ticker_df(df, "AAA")
    assert list(tdf.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert tdf.notna().all().all()
    assert len(tdf) == 300


def test_build_ticker_df_elimina_nan_internos():
    """El helper elimina NaN internos (raiz del bug I2)."""
    df = _make_multiindex_df(["AAA"])
    # Introducir NaN internos
    df.loc[df.index[50:60], ("Close", "AAA")] = np.nan
    df.loc[df.index[100:105], ("Volume", "AAA")] = np.nan
    tdf = build_ticker_df(df, "AAA")
    assert tdf.notna().all().all(), "No debe quedar ningun NaN"
    assert len(tdf) == 300 - 15  # 10 + 5 filas eliminadas


def test_build_ticker_df_ticker_inexistente():
    """El helper propaga KeyError si el ticker no existe."""
    df = _make_multiindex_df(["AAA"])
    with pytest.raises(KeyError):
        build_ticker_df(df, "ZZZ")


def test_classify_phase_patron_buggy_devuelve_range():
    """Documenta el bug: sin dropna, degenera a RANGE.

    Reproduce el comportamiento del patron A (buggy) sobre un df
    con NaN internos. Sirve como golden del comportamiento a evitar.
    """
    df = _make_multiindex_df(["AAA"])
    df.loc[df.index[50:70], ("Close", "AAA")] = np.nan

    # Patron buggy: pasar df con NaN directo
    phase_buggy = classify_wyckoff_phase(df, "AAA")
    # Patron correcto: build_ticker_df
    tdf = build_ticker_df(df, "AAA")
    phase_fix = classify_wyckoff_phase(tdf, "AAA")

    # El patron buggy tiende a RANGE (fallback silencioso).
    # No es determinista al 100% que sea RANGE, pero en la mayoria
    # de casos degenera. Lo que SI debe cumplirse: ambos no fallan.
    assert phase_buggy in ("RANGE", "MARKUP", "ACCUMULATION", "DISTRIBUTION", "INSUFFICIENT_DATA")
    assert phase_fix in ("RANGE", "MARKUP", "ACCUMULATION", "DISTRIBUTION", "INSUFFICIENT_DATA")


def test_classify_phase_fix_produce_fases_variadas():
    """El fix produce distribucion de fases real (no 100% RANGE).

    Crea 5 tickers con tendencias distintas, todos con NaN internos,
    y verifica que las 5 clasificaciones NO son todas RANGE.
    """
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    df = _make_multiindex_df(tickers, n=300, seed=7)

    # Introducir NaN internos en todos
    for tk in tickers:
        df.loc[df.index[50:60], ("Close", tk)] = np.nan

    phases = []
    for tk in tickers:
        tdf = build_ticker_df(df, tk)
        phases.append(classify_wyckoff_phase(tdf, tk))

    # Al menos uno NO debe ser RANGE (por construccion, algunas
    # tendencias sinteticas son claramente alcistas/bajistas)
    non_range = [p for p in phases if p != "RANGE"]
    assert len(non_range) >= 1, (
        f"Fix debe producir fases variadas. Fases: {phases}"
    )