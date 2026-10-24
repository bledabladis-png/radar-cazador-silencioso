from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.wyckoff_range import detect_ar, WyckoffEvent


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


def _bc_event(idx, high_val=101.0, date_pos=30):
    return WyckoffEvent(
        ticker="TEST", event_type="BC",
        event_date=idx[date_pos], available_at=idx[date_pos],
        status="CONFIRMED", evidence={"high": high_val},
    )


def test_ar_not_evaluable_without_confirmed_bc():
    n = 60
    df = _make_df(
        close=np.full(n, 100.0),
        high=np.full(n, 101.0),
        low=np.full(n, 99.0),
        volume=np.full(n, 1e6),
    )
    bc = WyckoffEvent(ticker="TEST", event_type="BC",
                      event_date=None, available_at=None,
                      status="NOT_OBSERVED")
    ev = detect_ar(df, "TEST", bc)
    assert ev.status == "NOT_EVALUABLE"
    assert ev.evidence.get("reason") == "bc_not_confirmed"


def test_ar_not_observed_if_drop_below_threshold():
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)  # ~2% drop, umbral 5%
    df = _make_df(close, high, low, np.full(n, 1e6))
    bc = _bc_event(idx, high_val=101.0, date_pos=30)
    ev = detect_ar(df, "TEST", bc, min_drop_pct=0.05)
    assert ev.status == "NOT_OBSERVED"
    assert ev.event_date is None
    assert "drop_below_threshold" in ev.evidence.get("reason", "")


def test_ar_candidate_if_drop_no_bounce():
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    low[35] = 85.0  # ~16% drop
    close[35:] = 85.5  # <3% recovery desde 85
    df = _make_df(close, high, low, np.full(n, 1e6))
    bc = _bc_event(idx, high_val=101.0, date_pos=30)
    ev = detect_ar(df, "TEST", bc, min_drop_pct=0.05, min_bounce_pct=0.03)
    assert ev.status == "CANDIDATE"
    assert ev.event_date == idx[35]
    assert ev.available_at == idx[35]
    assert "support" not in ev.evidence


def test_ar_confirmed_on_drop_and_bounce():
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    low[35] = 85.0
    close[35:40] = 85.5  # <3%
    close[40:] = 90.0  # rebote ~5.9% desde 85
    df = _make_df(close, high, low, np.full(n, 1e6))
    bc = _bc_event(idx, high_val=101.0, date_pos=30)
    ev = detect_ar(df, "TEST", bc, min_drop_pct=0.05, min_bounce_pct=0.03)
    assert ev.status == "CONFIRMED"
    assert ev.event_date == idx[35]
    assert ev.available_at == idx[40]
    assert ev.available_at > ev.event_date
    assert ev.evidence["support"] == 85.0
    assert ev.evidence["bounce_pct"] > 0.03