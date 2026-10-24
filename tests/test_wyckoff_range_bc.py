from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.wyckoff_distribution import detect_bc, WyckoffEvent, MIN_OBS_BC


def _make_df(close, high, low, volume, open_=None, ticker="TEST"):
    n = len(close)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    o = np.asarray(open_, dtype=float) if open_ is not None else np.asarray(close, dtype=float)
    df = pd.DataFrame(
        {
            ("Open", ticker): o,
            ("High", ticker): np.asarray(high, dtype=float),
            ("Low", ticker): np.asarray(low, dtype=float),
            ("Close", ticker): np.asarray(close, dtype=float),
            ("Volume", ticker): np.asarray(volume, dtype=float),
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "ticker"])
    return df


def test_bc_not_evaluable_on_short_series():
    n = 10
    df = _make_df(
        close=np.linspace(10, 11, n),
        high=np.linspace(10.1, 11.1, n),
        low=np.linspace(9.9, 10.9, n),
        volume=np.full(n, 1e6),
    )
    ev = detect_bc(df, "TEST")
    assert isinstance(ev, WyckoffEvent)
    assert ev.status == "NOT_EVALUABLE"
    assert ev.event_date is None
    assert ev.evidence.get("reason") == "insufficient_data"


def test_bc_not_evaluable_on_flat_series():
    n = MIN_OBS_BC + 20
    df = _make_df(
        close=np.full(n, 100.0),
        high=np.full(n, 100.5),
        low=np.full(n, 99.5),
        volume=np.full(n, 1e6),
    )
    ev = detect_bc(df, "TEST")
    assert ev.status == "NOT_EVALUABLE"


def test_bc_not_observed_on_oscillating_series_no_climax():
    n = 300  # > WYCKOFF_TREND_SLOW_MA(200) + robust_zscore(200) margen
    rng = np.random.default_rng(11)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.3, n))
    high = close + 0.3
    low = close - 0.3
    volume = np.full(n, 1e6) + rng.normal(0, 5e4, n)
    df = _make_df(close=close, high=high, low=low, volume=volume)
    ev = detect_bc(df, "TEST", lookback=30, v_threshold=5.0,
                   c_threshold=5.0, t_threshold=5.0)
    assert ev.status == "NOT_OBSERVED", f"got {ev.status}, evidence={ev.evidence}"
    assert ev.event_date is None


def test_bc_confirmed_at_clear_climax():
    n = 300
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.3, n))
    high = close + 0.3
    low = close - 0.3
    volume = np.full(n, 1e6) + rng.normal(0, 5e4, n)

    high[-1] = high.max() + 5.0
    low[-1] = low[-1] - 3.0
    close[-1] = high[-1] - 0.2
    volume[-1] = volume.mean() * 8.0

    df = _make_df(close=close, high=high, low=low, volume=volume)
    ev = detect_bc(df, "TEST", lookback=30, v_threshold=0.3,
                   c_threshold=0.5, t_threshold=-1.0)
    assert ev.status == "CONFIRMED", f"got {ev.status}, evidence={ev.evidence}"
    assert ev.event_date == df.index[-1]
    assert ev.available_at == df.index[-1]
    assert ev.evidence["is_max"] is True
    assert ev.evidence["is_vol"] is True
    assert ev.evidence["is_volat"] is True