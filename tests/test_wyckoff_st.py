"""Tests de ST (Secondary Test) en ambas cascadas."""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.wyckoff_distribution import detect_st_dist
from indicators.wyckoff_accumulation import detect_st_acc
from indicators.wyckoff_event import WyckoffEvent


def _make_df(close, high, low, volume, ticker="TEST"):
    n = len(close)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            ("Open", ticker): np.asarray(close, dtype=float),
            ("High", ticker): np.asarray(high, dtype=float),
            ("Low", ticker): np.asarray(low, dtype=float),
            ("Close", ticker): np.asarray(close, dtype=float),
            ("Volume", ticker): np.asarray(volume, dtype=float),
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "ticker"])
    return df


def _mk_bc(ticker, idx, pos, high_val):
    return WyckoffEvent(
        ticker=ticker, event_type="BC",
        event_date=idx[pos], available_at=idx[pos],
        status="CONFIRMED", evidence={"high": high_val},
    )


def _mk_sc(ticker, idx, pos, low_val):
    return WyckoffEvent(
        ticker=ticker, event_type="SC",
        event_date=idx[pos], available_at=idx[pos],
        status="CONFIRMED", evidence={"low": low_val},
    )


# --- ST en distribucion ---

def test_st_dist_not_evaluable_without_bc():
    n = 20
    df = _make_df(np.full(n, 100.0), np.full(n, 101.0),
                  np.full(n, 99.0), np.full(n, 1e6))
    bc = WyckoffEvent(ticker="TEST", event_type="BC",
                      event_date=None, available_at=None,
                      status="NOT_OBSERVED")
    ev = detect_st_dist(df, "TEST", bc)
    assert ev.status == "NOT_EVALUABLE"
    assert ev.evidence["reason"] == "bc_not_confirmed"


def test_st_dist_confirmed_at_retest():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 95.0)
    high = np.full(n, 96.0)
    low = np.full(n, 94.0)
    # BC en idx[5] con high=100
    # ST en idx[15]: high cerca de 100, close bajo rejection (100*0.95=95)
    high[15] = 99.0       # >= 100*0.97 = 97
    close[15] = 93.0      # < 95
    df = _make_df(close, high, low, np.full(n, 1e6))
    bc = _mk_bc("TEST", idx, 5, 100.0)
    ev = detect_st_dist(df, "TEST", bc)
    assert ev.status == "CONFIRMED"
    assert ev.event_date == idx[15]


def test_st_dist_not_observed_if_no_retest():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 80.0)  # nunca se acerca a 100
    high = np.full(n, 81.0)
    low = np.full(n, 79.0)
    df = _make_df(close, high, low, np.full(n, 1e6))
    bc = _mk_bc("TEST", idx, 5, 100.0)
    ev = detect_st_dist(df, "TEST", bc)
    assert ev.status == "NOT_OBSERVED"
    assert ev.evidence["reason"] == "no_st_in_window"


def test_st_dist_not_confirmed_if_close_not_rejected():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 95.0)
    high = np.full(n, 96.0)
    low = np.full(n, 94.0)
    # Retest pero cierre fuerte (no rechazo)
    high[15] = 99.0
    close[15] = 98.0  # cerca de 100, no rechazo
    df = _make_df(close, high, low, np.full(n, 1e6))
    bc = _mk_bc("TEST", idx, 5, 100.0)
    ev = detect_st_dist(df, "TEST", bc)
    assert ev.status == "NOT_OBSERVED"


# --- ST en acumulacion ---

def test_st_acc_not_evaluable_without_sc():
    n = 20
    df = _make_df(np.full(n, 100.0), np.full(n, 101.0),
                  np.full(n, 99.0), np.full(n, 1e6))
    sc = WyckoffEvent(ticker="TEST", event_type="SC",
                      event_date=None, available_at=None,
                      status="NOT_OBSERVED")
    ev = detect_st_acc(df, "TEST", sc)
    assert ev.status == "NOT_EVALUABLE"
    assert ev.evidence["reason"] == "sc_not_confirmed"


def test_st_acc_confirmed_at_retest():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 105.0)
    high = np.full(n, 106.0)
    low = np.full(n, 104.0)
    # SC en idx[5] con low=100
    # ST en idx[15]: low cerca de 100, close por encima de 105
    low[15] = 101.0        # <= 100*1.03 = 103
    close[15] = 106.0      # > 100*1.05 = 105
    df = _make_df(close, high, low, np.full(n, 1e6))
    sc = _mk_sc("TEST", idx, 5, 100.0)
    ev = detect_st_acc(df, "TEST", sc)
    assert ev.status == "CONFIRMED"
    assert ev.event_date == idx[15]


def test_st_acc_not_observed_if_no_retest():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 120.0)  # se aleja de 100
    high = np.full(n, 121.0)
    low = np.full(n, 119.0)
    df = _make_df(close, high, low, np.full(n, 1e6))
    sc = _mk_sc("TEST", idx, 5, 100.0)
    ev = detect_st_acc(df, "TEST", sc)
    assert ev.status == "NOT_OBSERVED"
    assert ev.evidence["reason"] == "no_st_in_window"


def test_st_acc_not_confirmed_if_close_not_rejected():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 105.0)
    high = np.full(n, 106.0)
    low = np.full(n, 104.0)
    low[15] = 101.0
    close[15] = 102.0  # cierra bajo rechazo, no rechazo real
    df = _make_df(close, high, low, np.full(n, 1e6))
    sc = _mk_sc("TEST", idx, 5, 100.0)
    ev = detect_st_acc(df, "TEST", sc)
    assert ev.status == "NOT_OBSERVED"