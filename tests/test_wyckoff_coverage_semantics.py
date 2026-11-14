# -*- coding: utf-8 -*-
"""FIX-COB-WYCKOFF (2026-11-14): coverage_data_pct vs coverage_weight_pct."""
import pandas as pd
import pytest

from indicators import wyckoff_holdings_report as whr


def _make_df_with_n_bars(tickers_with_bars):
    n_total = max(max(tickers_with_bars.values()), 300)
    dates = pd.date_range("2023-01-01", periods=n_total, freq="B")
    tickers = list(tickers_with_bars.keys())
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], tickers])
    df = pd.DataFrame(100.0, index=dates, columns=cols)
    for t, nb in tickers_with_bars.items():
        start = n_total - nb
        for field in ("Open", "High", "Low", "Close", "Volume"):
            df.loc[df.index[:start], (field, t)] = float("nan")
    return df


class _FPStub:
    phase = "ACCUMULATION"
    event_phase = "ACCUMULATION"
    phase_since = "2026-01-01"
    current_phase_basis = ""


def test_coverage_data_pct_refleja_datos_no_seleccion(monkeypatch):
    """10 tickers, 5 con datos >=200, 5 con 10. weight_pct=100, data_pct=50."""
    tickers = [f"T{i:02d}" for i in range(10)]
    bars = {t: (300 if i < 5 else 10) for i, t in enumerate(tickers)}
    df = _make_df_with_n_bars(bars)
    monkeypatch.setattr(whr, "classify_phase", lambda *a, **k: _FPStub())
    holdings = pd.DataFrame({
        "etf": ["XLK"] * 10, "ticker": tickers, "weight": [10.0] * 10,
    })
    snap = whr.compute_wyckoff_snapshot(
        df, holdings, None, None, selection_mode="target_weight")
    assert len(snap) == 5
    assert snap["coverage_weight_pct"].iloc[0] == pytest.approx(100.0)
    assert snap["coverage_data_pct"].iloc[0] == pytest.approx(50.0)


def test_coverage_data_pct_igual_a_weight_si_todos_con_datos(monkeypatch):
    tickers = ["A", "B", "C"]
    df = _make_df_with_n_bars({t: 300 for t in tickers})
    monkeypatch.setattr(whr, "classify_phase", lambda *a, **k: _FPStub())
    holdings = pd.DataFrame({
        "etf": ["XLK"] * 3, "ticker": tickers,
        "weight": [33.33, 33.33, 33.34],
    })
    snap = whr.compute_wyckoff_snapshot(
        df, holdings, None, None, selection_mode="target_weight")
    assert len(snap) == 3
    assert snap["coverage_weight_pct"].iloc[0] == pytest.approx(
        snap["coverage_data_pct"].iloc[0], rel=1e-6)
