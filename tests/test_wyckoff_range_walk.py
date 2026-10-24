from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from indicators.wyckoff_distribution import walk_distribution


def _make_df(n, ticker="TEST"):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.linspace(100.0, 110.0, n)
    df = pd.DataFrame(
        {
            ("Open", ticker): close,
            ("High", ticker): close + 0.5,
            ("Low", ticker): close - 0.5,
            ("Close", ticker): close,
            ("Volume", ticker): np.full(n, 1e6),
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "ticker"])
    return df


def test_walk_returns_dataframe_with_schema():
    df = _make_df(20)
    out = walk_distribution(df, "TEST")
    assert isinstance(out, pd.DataFrame)
    for col in ("date", "classification", "structural_state", "range_id",
                "range_type", "trend_direction", "sow_status",
                "coherence_error", "is_first", "is_transition"):
        assert col in out.columns, f"falta {col}"


def test_walk_first_row_marked_first():
    df = _make_df(20)
    out = walk_distribution(df, "TEST")
    if len(out) > 0:
        assert bool(out.iloc[0]["is_first"]) is True


def test_walk_rejects_invalid_start():
    df = _make_df(20)
    with pytest.raises(ValueError):
        walk_distribution(df, "TEST", start=pd.Timestamp("1999-01-01"))


def test_walk_rejects_end_before_start():
    df = _make_df(20)
    with pytest.raises(ValueError):
        walk_distribution(df, "TEST", start=df.index[-1], end=df.index[0])


def test_walk_flat_series_single_transition():
    """Serie corta y uniforme: solo la primera fila, sin transiciones
    posteriores."""
    df = _make_df(20)
    out = walk_distribution(df, "TEST")
    # Con 20 barras y sin BC, todas las barras dan NOT_EVALUABLE por
    # insufficient_data. Solo la primera fila se emite.
    if len(out) > 1:
        # Si hay >1, debe ser porque hubo transicion real de clasificacion
        assert bool(out.iloc[0]["is_first"]) is True
        assert out["classification"].nunique() > 1


@pytest.mark.integration_slow
def test_walk_acs_mc_reaches_distribution_confirmed():
    """Cascada completa sobre ACS.MC: en algun momento debe emitir
    DISTRIBUTION_CONFIRMED, y la primera fila debe ser NOT_EVALUABLE."""
    p = Path("data/stock_prices_extended.parquet")
    if not p.exists():
        pytest.skip("parquet no disponible")
    df = pd.read_parquet(p)
    ticker = "ACS.MC"
    if ticker not in df.columns.get_level_values(1):
        pytest.skip(f"{ticker} no en parquet")

    close = df[("Close", ticker)].dropna()
    # Empezar en bar 250 para evitar warmup y acelerar
    start = close.index[250]
    end = close.index[-1]

    out = walk_distribution(df, ticker, start=start, end=end)
    assert len(out) > 0, "walk no emitio filas"
    assert bool(out.iloc[0]["is_first"]) is True

    # Debe haber alcanzado DISTRIBUTION_CONFIRMED en algun momento
    assert "DISTRIBUTION_CONFIRMED" in out["classification"].values, (
        f"no llego a DISTRIBUTION_CONFIRMED. clasificaciones: "
        f"{out['classification'].tolist()}"
    )
    # La ultima fila debe ser DISTRIBUTION_CONFIRMED (bar actual)
    assert out.iloc[-1]["classification"] == "DISTRIBUTION_CONFIRMED"