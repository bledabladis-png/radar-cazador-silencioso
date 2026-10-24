from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.wyckoff_distribution import (
    Range, detect_sow_in_range,
)


def _make_df(closes, ticker="TEST"):
    n = len(closes)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            ("Open", ticker): np.asarray(closes, dtype=float),
            ("High", ticker): np.asarray(closes, dtype=float) + 0.5,
            ("Low", ticker): np.asarray(closes, dtype=float) - 0.5,
            ("Close", ticker): np.asarray(closes, dtype=float),
            ("Volume", ticker): np.full(n, 1e6),
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "ticker"])
    return df


def _make_active_range(idx, active_pos=10, support=100.0, resistance=120.0,
                       tol_frac=0.05):
    width = resistance - support
    tol_abs = tol_frac * width
    return Range(
        range_id="TEST_1d_2020-01-01_2020-01-10",
        ticker="TEST",
        timeframe="1d",
        formation_path="BC_AR",
        range_type="DISTRIBUTION_CANDIDATE",
        candidate_start_date=idx[0],
        start_available_at=idx[active_pos],
        range_active_at=idx[active_pos],
        support=support,
        resistance=resistance,
        tolerance_frac=tol_frac,
        tol_abs=tol_abs,
        state="RANGE_ACTIVE",
    )


def test_sow_not_evaluable_if_range_not_active():
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    df = _make_df(np.full(20, 100.0))
    r = _make_active_range(idx)
    r_broken = Range(**{**r.__dict__, "state": "RANGE_BROKEN"})
    ev = detect_sow_in_range(df, "TEST", r_broken)
    assert ev.status == "NOT_EVALUABLE"
    assert ev.evidence["reason"] == "range_not_active"


def test_sow_not_observed_if_price_stays_inside():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = _make_df(np.full(n, 110.0))
    r = _make_active_range(idx, active_pos=10)
    ev = detect_sow_in_range(df, "TEST", r)
    assert ev.status == "NOT_OBSERVED"
    assert ev.evidence["reason"] == "no_close_below_support_minus_tol"


def test_sow_confirmed_after_m_closes_under_support():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 110.0)
    close[15] = 94.0   # candidate (< 99 = support - tol)
    close[16] = 93.5
    close[17] = 93.0   # M=3 -> CONFIRMED at idx[17]
    df = _make_df(close)
    r = _make_active_range(idx, active_pos=10)
    ev = detect_sow_in_range(df, "TEST", r, confirm_closes=3, timeout_sessions=10)
    assert ev.status == "CONFIRMED"
    assert ev.event_date == idx[15]
    assert ev.available_at == idx[15]
    assert ev.evidence["resolution_event_date"] == str(idx[17])
    assert ev.evidence["closes_under"] == 3


def test_sow_rejected_if_price_recovers_before_m():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 110.0)
    close[15] = 94.0
    close[16] = 115.0  # > 101 = support + tol -> REJECTED
    df = _make_df(close)
    r = _make_active_range(idx, active_pos=10)
    ev = detect_sow_in_range(df, "TEST", r, confirm_closes=3, timeout_sessions=10)
    assert ev.status == "REJECTED"
    assert ev.event_date == idx[15]
    assert ev.evidence["resolution_event_date"] == str(idx[16])
    assert ev.evidence["reason"] == "close_above_support_plus_tol"


def test_sow_expired_if_timeout_without_confirm_or_reject():
    # support=100, tol_abs=1.0 -> upper_break=99, upper_recover=101
    # zona gris segura: 100.5 (entre 99 y 101)
    n = 40
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.5)
    close[15] = 94.0    # candidate (1 close under)
    # pos 16..20 = 100.5 (zona gris, ni confirma ni rechaza)
    df = _make_df(close)
    r = _make_active_range(idx, active_pos=10)
    ev = detect_sow_in_range(df, "TEST", r, confirm_closes=3, timeout_sessions=5)
    assert ev.status == "EXPIRED", f"got {ev.status}, evidence={ev.evidence}"
    assert ev.event_date == idx[15]
    assert ev.evidence["closes_under"] == 1
    assert "resolution_event_date" in ev.evidence


def test_sow_candidate_if_still_pending_as_of():
    # n=20, timeout=20 -> timeout_pos=min(35, 19)=19, as_of=19 < 15+20
    # necesita zona gris para no rechazar antes de pos 19
    n = 20
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.5)
    close[15] = 94.0
    df = _make_df(close)
    r = _make_active_range(idx, active_pos=10)
    ev = detect_sow_in_range(df, "TEST", r, confirm_closes=3, timeout_sessions=20)
    assert ev.status == "CANDIDATE", f"got {ev.status}, evidence={ev.evidence}"
    assert ev.event_date == idx[15]
    assert ev.evidence["reason"] == "pending_as_of"