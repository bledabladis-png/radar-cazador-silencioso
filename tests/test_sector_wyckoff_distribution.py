"""Tests de compute_sector_wyckoff_distribution.

F4-06b, F4-07, F4-08 (2026-09-28): reescritos. Los tests originales
eran:
  - test_pct_sum_100: no-op (`pass`) con comentario de no aplica.
  - test_coverage_formula: reimplementaba la formula localmente sin
    llamar a la funcion real.
  - test_insufficient_valid_returns_nan: assert tautologico
    (`pd.isna(np.nan)`).

Los tests nuevos llaman a compute_sector_wyckoff_distribution con
mocks de build_ticker_df y classify_wyckoff_phase.
"""
import pandas as pd
import pytest


def _make_df_stocks(tickers, n_rows=70):
    """DataFrame MultiIndex (Close, ticker) con DatetimeIndex."""
    dates = pd.date_range("2026-01-01", periods=n_rows, freq="D")
    cols = pd.MultiIndex.from_product([["Close"], tickers])
    return pd.DataFrame(100.0, index=dates, columns=cols)


def _make_holdings(tickers, etf="XLK"):
    return pd.DataFrame({"etf": [etf] * len(tickers), "ticker": list(tickers)})


def _patch_chain(monkeypatch, phase="ACCUMULATION"):
    from indicators import sector_wyckoff_distribution as swd
    monkeypatch.setattr(swd, "build_ticker_df", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(swd, "classify_wyckoff_phase",
                        lambda *a, **k: phase)


def test_coverage_y_pct_calculados_por_la_funcion(monkeypatch):
    """F4-08: la cobertura y los pct_* los calcula la funcion real.

    6 tickers validos -> n_valid=6 >= 5 -> pct_accumulation=100, resto=0.
    """
    from indicators import sector_wyckoff_distribution as swd
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    df_stocks = _make_df_stocks(tickers)
    holdings = _make_holdings(tickers)
    _patch_chain(monkeypatch, phase="ACCUMULATION")

    out = swd.compute_sector_wyckoff_distribution(df_stocks, holdings)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["sector"] == "XLK"
    assert row["n_total"] == 6
    assert row["n_valid_wyckoff"] == 6
    assert row["n_insufficient_wyckoff"] == 0
    assert row["coverage_wyckoff"] == pytest.approx(100.0)
    assert row["pct_accumulation"] == pytest.approx(100.0)
    for p in ["markup", "range", "distribution", "markdown"]:
        assert row[f"pct_{p}"] == pytest.approx(0.0)


def test_pct_nan_cuando_n_valid_menor_que_5(monkeypatch):
    """F4-07: con n_valid < 5, los pct_* son NaN (no assert tautologico)."""
    from indicators import sector_wyckoff_distribution as swd
    tickers = ["AAPL", "MSFT", "GOOGL"]
    df_stocks = _make_df_stocks(tickers)
    holdings = _make_holdings(tickers)
    _patch_chain(monkeypatch, phase="MARKUP")

    out = swd.compute_sector_wyckoff_distribution(df_stocks, holdings)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_valid_wyckoff"] == 3
    assert row["coverage_wyckoff"] == pytest.approx(100.0)
    for p in ["accumulation", "markup", "range", "distribution", "markdown"]:
        assert pd.isna(row[f"pct_{p}"])
    assert row["count_markup"] == 3
