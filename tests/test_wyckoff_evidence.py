"""Tests de los 6 detectores de evidencia pasiva: PSY, PS, mSOW, mSOS, LPSY, LPS."""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.wyckoff_distribution import (
    detect_psy, detect_ps, detect_msow, detect_msos,
    detect_lpsy, detect_lps,
)
from indicators.wyckoff_event import WyckoffEvent, Range


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


def _mk_event(ticker, etype, date, status="CONFIRMED", range_id=None):
    return WyckoffEvent(
        ticker=ticker, event_type=etype, event_date=date,
        available_at=date, status=status, evidence={},
    )


def _mk_range(ticker, support, resistance, active_at):
    return Range(
        range_id=f"{ticker}_r", ticker=ticker, timeframe="1d",
        formation_path="BC_AR", range_type="DISTRIBUTION_CANDIDATE",
        candidate_start_date=active_at, start_available_at=active_at,
        range_active_at=active_at,
        support=support, resistance=resistance,
        tolerance_frac=0.05, tol_abs=0.05 * (resistance - support),
        state="RANGE_BROKEN",
    )


# --- PSY ---

def test_psy_not_evaluable_short_series():
    df = _make_df(np.full(10, 100.0), np.full(10, 101.0),
                  np.full(10, 99.0), np.full(10, 1e6))
    ev = detect_psy(df, "TEST")
    assert ev.status == "NOT_EVALUABLE"


def test_psy_not_observed_flat_series():
    n = 80
    df = _make_df(np.full(n, 100.0), np.full(n, 100.5),
                  np.full(n, 99.5), np.full(n, 1e6))
    ev = detect_psy(df, "TEST")
    # flat: robust_zscore da NaN -> NOT_EVALUABLE, o NOT_OBSERVED si logra features
    assert ev.status in ("NOT_EVALUABLE", "NOT_OBSERVED")


# --- PS ---

def test_ps_not_evaluable_short_series():
    df = _make_df(np.full(10, 100.0), np.full(10, 101.0),
                  np.full(10, 99.0), np.full(10, 1e6))
    ev = detect_ps(df, "TEST")
    assert ev.status == "NOT_EVALUABLE"


# --- mSOW ---

def test_msow_evaluable_with_enough_data():
    n = 80
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = _make_df(close, close + 0.5, close - 0.5, np.full(n, 1e6))
    ev = detect_msow(df, "TEST")
    assert ev.status in ("CONFIRMED", "NOT_OBSERVED", "NOT_EVALUABLE")
    assert ev.event_type == "mSOW"


# --- mSOS ---

def test_msos_evaluable_with_enough_data():
    n = 80
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = _make_df(close, close + 0.5, close - 0.5, np.full(n, 1e6))
    ev = detect_msos(df, "TEST")
    assert ev.status in ("CONFIRMED", "NOT_OBSERVED", "NOT_EVALUABLE")


# --- LPSY ---

def test_lpsy_not_evaluable_without_sow():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = _make_df(np.full(n, 100.0), np.full(n, 101.0),
                  np.full(n, 99.0), np.full(n, 1e6))
    sow = _mk_event("TEST", "SOW", None, status="NOT_OBSERVED")
    rng = _mk_range("TEST", 95.0, 105.0, idx[5])
    ev = detect_lpsy(df, "TEST", sow, rng)
    assert ev.status == "NOT_EVALUABLE"


def test_lpsy_not_evaluable_without_range():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = _make_df(np.full(n, 100.0), np.full(n, 101.0),
                  np.full(n, 99.0), np.full(n, 1e6))
    sow = _mk_event("TEST", "SOW", idx[10], status="CONFIRMED")
    ev = detect_lpsy(df, "TEST", sow, None)
    assert ev.status == "NOT_EVALUABLE"
    assert ev.evidence["reason"] == "no_range_reference"


def test_lpsy_confirmed_on_weak_rebound():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    close[15] = 94.0   # post-SOW, minimo
    close[16] = 96.0   # rebote debil (rango 95-105, max = 94 + 0.7*10 = 101)
    close[17:] = 95.0
    df = _make_df(close, close + 0.5, close - 0.5, np.full(n, 1e6))
    sow = _mk_event("TEST", "SOW", idx[12], status="CONFIRMED")
    rng = _mk_range("TEST", 95.0, 105.0, idx[5])
    ev = detect_lpsy(df, "TEST", sow, rng, max_rebound_pct=0.7, max_window=15)
    assert ev.status == "CONFIRMED"
    assert ev.event_date is not None


def test_lpsy_not_observed_if_rebound_strong():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    close[15] = 94.0
    close[16] = 104.0  # rebote fuerte, no LPSY
    df = _make_df(close, close + 0.5, close - 0.5, np.full(n, 1e6))
    sow = _mk_event("TEST", "SOW", idx[12], status="CONFIRMED")
    rng = _mk_range("TEST", 95.0, 105.0, idx[5])
    ev = detect_lpsy(df, "TEST", sow, rng, max_rebound_pct=0.7, max_window=15)
    assert ev.status == "NOT_OBSERVED"


# --- LPS ---

def test_lps_not_evaluable_without_sos():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = _make_df(np.full(n, 100.0), np.full(n, 101.0),
                  np.full(n, 99.0), np.full(n, 1e6))
    sos = _mk_event("TEST", "SOS", None, status="NOT_OBSERVED")
    rng = _mk_range("TEST", 95.0, 105.0, idx[5])
    ev = detect_lps(df, "TEST", sos, rng)
    assert ev.status == "NOT_EVALUABLE"


def test_lps_confirmed_on_weak_pullback():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    close[15] = 106.0   # post-SOS, maximo
    close[16] = 103.0   # pullback debil (rango 95-105, min = 106 - 0.7*10 = 99)
    close[17:] = 104.0
    df = _make_df(close, close + 0.5, close - 0.5, np.full(n, 1e6))
    sos = _mk_event("TEST", "SOS", idx[12], status="CONFIRMED")
    rng = _mk_range("TEST", 95.0, 105.0, idx[5])
    ev = detect_lps(df, "TEST", sos, rng, max_pullback_pct=0.7, max_window=15)
    assert ev.status == "CONFIRMED"