"""Tests de la cascada de acumulacion: SC, AR_ACCUM, Range, SOS."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.wyckoff_accumulation import (
    detect_sc, detect_ar_accumulation, build_range_from_sc_ar,
    detect_sos_in_range, classify_accumulation,
)
from indicators.wyckoff_event import WyckoffEvent, Range


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


def _mk_event(ticker, etype, event_date, available_at, status="CONFIRMED",
              evidence=None, range_id=None):
    return WyckoffEvent(
        ticker=ticker, event_type=etype,
        event_date=event_date, available_at=available_at,
        status=status, evidence=evidence or {}, range_id=range_id,
    )


# --- SC ---

def test_sc_not_evaluable_on_short_series():
    n = 10
    df = _make_df(
        close=np.linspace(10, 11, n), high=np.linspace(10.1, 11.1, n),
        low=np.linspace(9.9, 10.9, n), volume=np.full(n, 1e6),
    )
    ev = detect_sc(df, "TEST")
    assert ev.status == "NOT_EVALUABLE"
    assert ev.evidence.get("reason") == "insufficient_data"


def test_sc_not_observed_no_minimum():
    n = 300
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.3, n))
    high = close + 0.3
    low = close - 0.3
    volume = np.full(n, 1e6) + rng.normal(0, 5e4, n)
    df = _make_df(close=close, high=high, low=low, volume=volume)
    ev = detect_sc(df, "TEST", lookback=30, v_threshold=5.0,
                   c_threshold=5.0, t_threshold=-5.0)
    assert ev.status == "NOT_OBSERVED"


def test_sc_confirmed_at_clear_climax():
    n = 300
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(-0.05, 0.3, n))
    high = close + 0.3
    low = close - 0.3
    volume = np.full(n, 1e6) + rng.normal(0, 5e4, n)

    low[-1] = low.min() - 5.0
    high[-1] = high[-1] + 3.0
    close[-1] = low[-1] + 0.2
    volume[-1] = volume.mean() * 8.0

    df = _make_df(close=close, high=high, low=low, volume=volume)
    ev = detect_sc(df, "TEST", lookback=30, v_threshold=0.3,
                   c_threshold=0.5, t_threshold=1.0)
    assert ev.status == "CONFIRMED", f"got {ev.status}, evidence={ev.evidence}"
    assert ev.event_date == df.index[-1]
    assert ev.available_at == df.index[-1]


# --- AR_ACCUM ---

def _sc_event(idx, low_val=100.0, date_pos=30):
    return _mk_event("TEST", "SC", idx[date_pos], idx[date_pos],
                     evidence={"low": low_val})


def test_ar_accum_not_evaluable_without_sc():
    n = 60
    df = _make_df(np.full(n, 100.0), np.full(n, 101.0),
                  np.full(n, 99.0), np.full(n, 1e6))
    sc = _mk_event("TEST", "SC", None, None, status="NOT_OBSERVED")
    ev = detect_ar_accumulation(df, "TEST", sc)
    assert ev.status == "NOT_EVALUABLE"
    assert ev.evidence["reason"] == "sc_not_confirmed"


def test_ar_accum_not_observed_if_bounce_too_small():
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 101.0)
    high = np.full(n, 101.5)
    low = np.full(n, 99.0)
    df = _make_df(close, high, low, np.full(n, 1e6))
    sc = _sc_event(idx, low_val=100.0, date_pos=30)
    ev = detect_ar_accumulation(df, "TEST", sc, min_bounce_pct=0.05)
    assert ev.status == "NOT_OBSERVED"
    assert "bounce_below" in ev.evidence["reason"]


def test_ar_accum_candidate_without_pullback():
    # Fix: high[35] es el maximo unico (no todos 120.0)
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 115.0)
    high = np.full(n, 115.0)
    high[35] = 120.0  # maximo unico en idx[35]
    close[35:] = 119.5  # pullback < 3% desde 120
    low = np.full(n, 99.0)
    df = _make_df(close, high, low, np.full(n, 1e6))
    sc = _sc_event(idx, low_val=100.0, date_pos=30)
    ev = detect_ar_accumulation(df, "TEST", sc, min_bounce_pct=0.05,
                                 min_pullback_pct=0.03)
    assert ev.status == "CANDIDATE", f"got {ev.status}, evidence={ev.evidence}"
    assert ev.event_date == idx[35]


def test_ar_accum_confirmed_with_pullback():
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 115.0)
    high = np.full(n, 115.0)
    high[35] = 120.0
    close[35] = 119.5  # dia del high: sin pullback (0.4% < 3%)
    close[36:] = 115.0  # dia siguiente: pullback 4.2% desde 120
    low = np.full(n, 99.0)
    df = _make_df(close, high, low, np.full(n, 1e6))
    sc = _sc_event(idx, low_val=100.0, date_pos=30)
    ev = detect_ar_accumulation(df, "TEST", sc, min_bounce_pct=0.05,
                                 min_pullback_pct=0.03)
    assert ev.status == "CONFIRMED", f"got {ev.status}, evidence={ev.evidence}"
    assert ev.evidence["resistance"] == 120.0
    assert ev.event_date == idx[35]
    assert ev.available_at == idx[36]
    assert ev.available_at > ev.event_date


# --- Range ---

def test_build_range_from_sc_ar_active():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    sc = _mk_event("TEST", "SC", idx[2], idx[2], evidence={"low": 100.0})
    ar = _mk_event("TEST", "AR_ACCUM", idx[5], idx[7],
                   evidence={"resistance": 130.0})
    df = pd.DataFrame(
        {("Close", "TEST"): np.full(10, 115.0)}, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "ticker"])
    r = build_range_from_sc_ar(df, "TEST", sc, ar,
                                range_type="ACCUMULATION_CANDIDATE")
    assert r.state == "RANGE_ACTIVE"
    assert r.support == 100.0
    assert r.resistance == 130.0
    assert r.range_type == "ACCUMULATION_CANDIDATE"
    assert r.range_active_at == idx[7]


def test_build_range_from_sc_ar_raises_if_support_ge_resistance():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    sc = _mk_event("TEST", "SC", idx[2], idx[2], evidence={"low": 130.0})
    ar = _mk_event("TEST", "AR_ACCUM", idx[5], idx[7],
                   evidence={"resistance": 100.0})
    df = pd.DataFrame(
        {("Close", "TEST"): np.full(10, 115.0)}, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "ticker"])
    with pytest.raises(ValueError, match="support"):
        build_range_from_sc_ar(df, "TEST", sc, ar)


# --- SOS ---

def _active_acc_range(idx, active_pos=10, support=100.0, resistance=120.0,
                       tol_frac=0.05):
    width = resistance - support
    return Range(
        range_id="TEST_1d_ACC",
        ticker="TEST", timeframe="1d", formation_path="BC_AR",
        range_type="ACCUMULATION_CANDIDATE",
        candidate_start_date=idx[0], start_available_at=idx[active_pos],
        range_active_at=idx[active_pos],
        support=support, resistance=resistance,
        tolerance_frac=tol_frac, tol_abs=tol_frac * width,
        state="RANGE_ACTIVE",
    )


def test_sos_not_observed_inside_range():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = _make_df(np.full(n, 110.0), np.full(n, 110.5),
                  np.full(n, 109.5), np.full(n, 1e6))
    r = _active_acc_range(idx, active_pos=10)
    ev = detect_sos_in_range(df, "TEST", r)
    assert ev.status == "NOT_OBSERVED"
    assert "no_close_above" in ev.evidence["reason"]


def test_sos_confirmed_after_m_closes():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 110.0)
    close[15] = 127.0
    close[16] = 128.0
    close[17] = 129.0
    df = _make_df(close, close + 0.5, close - 0.5, np.full(n, 1e6))
    r = _active_acc_range(idx, active_pos=10)
    ev = detect_sos_in_range(df, "TEST", r, confirm_closes=3, timeout_sessions=10)
    assert ev.status == "CONFIRMED"
    assert ev.event_date == idx[15]
    assert ev.evidence["closes_above"] == 3


def test_sos_rejected_on_recovery():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 110.0)
    close[15] = 127.0
    close[16] = 115.0
    df = _make_df(close, close + 0.5, close - 0.5, np.full(n, 1e6))
    r = _active_acc_range(idx, active_pos=10)
    ev = detect_sos_in_range(df, "TEST", r, confirm_closes=3, timeout_sessions=10)
    assert ev.status == "REJECTED"
    assert ev.evidence["reason"] == "close_below_resistance_minus_tol"


# --- classify_accumulation ---

def test_classify_accumulation_not_evaluable_without_sc():
    n = 300
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.3, n))
    high = close + 0.3
    low = close - 0.3
    volume = np.full(n, 1e6) + rng.normal(0, 5e4, n)
    df = _make_df(close, high, low, volume)
    r = classify_accumulation(df, "TEST")
    assert r.classification == "NOT_EVALUABLE"
    assert r.evidence.get("reason") in ("sc_not_confirmed", "empty_df_up_to_as_of")


def test_classify_accumulation_wrapper_uses_last_bar():
    n = 30
    df = _make_df(np.full(n, 100.0), np.full(n, 100.5),
                  np.full(n, 99.5), np.full(n, 1e6))
    r = classify_accumulation(df, "TEST")
    assert r.as_of == df.index[-1]