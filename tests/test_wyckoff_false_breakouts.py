"""Tests de UT/UTAD y Spring (falsas rupturas)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.wyckoff_distribution import (
    detect_false_breakout_up, detect_false_breakout_down,
)
from indicators.wyckoff_event import Range


def _make_df(close, ticker="TEST"):
    n = len(close)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close_arr = np.asarray(close, dtype=float)
    df = pd.DataFrame(
        {
            ("Open", ticker): close_arr,
            ("High", ticker): close_arr + 0.5,
            ("Low", ticker): close_arr - 0.5,
            ("Close", ticker): close_arr,
            ("Volume", ticker): np.full(n, 1e6),
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "ticker"])
    return df


def _active_range(idx, support=95.0, resistance=105.0, tol_frac=0.02,
                   active_pos=5, state="RANGE_ACTIVE"):
    return Range(
        range_id="TEST_R",
        ticker="TEST", timeframe="1d",
        formation_path="BC_AR",
        range_type="DISTRIBUTION_CANDIDATE",
        candidate_start_date=idx[0],
        start_available_at=idx[active_pos],
        range_active_at=idx[active_pos],
        support=support, resistance=resistance,
        tolerance_frac=tol_frac, tol_abs=tol_frac * (resistance - support),
        state=state,
    )


# --- UT (falsa ruptura arriba) ---

def test_ut_not_evaluable_range_not_active():
    n = 20
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = _make_df(np.full(n, 100.0))
    r = _active_range(idx, state="RANGE_BROKEN")
    ev = detect_false_breakout_up(df, "TEST", r)
    assert ev.status == "NOT_EVALUABLE"
    assert ev.evidence["reason"] == "range_not_active"


def test_ut_not_observed_no_breakout():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)  # dentro del rango siempre
    df = _make_df(close)
    r = _active_range(idx, support=95.0, resistance=105.0, tol_frac=0.02,
                      active_pos=5)
    ev = detect_false_breakout_up(df, "TEST", r, max_window=10, confirm_sessions=2)
    assert ev.status == "NOT_OBSERVED"
    assert ev.evidence["reason"] == "no_breakout_above"


def test_ut_confirmed_at_false_breakout():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    # Ruptura arriba en idx[15]: 105 + 2% * 10 = 105.2, superado
    close[15] = 107.0
    # Retorno: 2 sesiones con cierre < 105
    close[16] = 103.0
    close[17] = 102.0
    df = _make_df(close)
    r = _active_range(idx, support=95.0, resistance=105.0, tol_frac=0.02,
                      active_pos=5)
    ev = detect_false_breakout_up(df, "TEST", r, max_window=10, confirm_sessions=2)
    assert ev.status == "CONFIRMED"
    assert ev.event_date == idx[15]
    assert ev.available_at == idx[17]


def test_ut_candidate_if_no_return_yet():
    n = 20
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    close[15] = 107.0  # ruptura
    close[16:] = 107.0  # todos por encima de resistance, sin retorno
    df = _make_df(close)
    r = _active_range(idx, support=95.0, resistance=105.0, tol_frac=0.02,
                      active_pos=5)
    ev = detect_false_breakout_up(df, "TEST", r, max_window=10, confirm_sessions=2)
    assert ev.status == "CANDIDATE"
    assert ev.event_date == idx[15]
    assert ev.available_at is None


# --- Spring (falsa ruptura abajo) ---

def test_spring_not_evaluable_range_not_active():
    n = 20
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = _make_df(np.full(n, 100.0))
    r = _active_range(idx, state="RANGE_EXPIRED")
    ev = detect_false_breakout_down(df, "TEST", r)
    assert ev.status == "NOT_EVALUABLE"
    assert ev.evidence["reason"] == "range_not_active"


def test_spring_not_observed_no_breakout():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    df = _make_df(close)
    r = _active_range(idx, support=95.0, resistance=105.0, tol_frac=0.02,
                      active_pos=5)
    ev = detect_false_breakout_down(df, "TEST", r, max_window=10, confirm_sessions=2)
    assert ev.status == "NOT_OBSERVED"
    assert ev.evidence["reason"] == "no_breakout_below"


def test_spring_confirmed_at_false_breakout():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    # Ruptura abajo en idx[15]: 95 - 2% * 10 = 94.8, superado
    close[15] = 92.0
    # Retorno: 2 sesiones con cierre > 95
    close[16] = 96.0
    close[17] = 97.0
    df = _make_df(close)
    r = _active_range(idx, support=95.0, resistance=105.0, tol_frac=0.02,
                      active_pos=5)
    ev = detect_false_breakout_down(df, "TEST", r, max_window=10, confirm_sessions=2)
    assert ev.status == "CONFIRMED"
    assert ev.event_date == idx[15]
    assert ev.available_at == idx[17]


def test_spring_candidate_if_no_recovery_yet():
    n = 20
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    close[15] = 92.0   # ruptura abajo
    close[16:] = 92.5  # todos por debajo de support, sin recuperacion
    df = _make_df(close)
    r = _active_range(idx, support=95.0, resistance=105.0, tol_frac=0.02,
                      active_pos=5)
    ev = detect_false_breakout_down(df, "TEST", r, max_window=10, confirm_sessions=2)
    assert ev.status == "CANDIDATE"
    assert ev.event_date == idx[15]
    assert ev.available_at is None