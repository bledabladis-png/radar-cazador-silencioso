# -*- coding: utf-8 -*-
"""Tests FU-018-3c: filtro EOD en lotes Yahoo (USA + UK)."""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.stock_data_loader import _filter_non_eod_batch


MADRID = ZoneInfo("Europe/Madrid")


def _make_batch(tickers, last_date="2026-09-15"):
    dates = pd.date_range(end=last_date, periods=3, freq="D")
    data = {}
    for t in tickers:
        for field in ("Open", "High", "Low", "Close", "Volume"):
            data[(field, t)] = np.arange(3, dtype=float) + 100
    return pd.DataFrame(data, index=dates)


def test_filter_empty_batch():
    df = pd.DataFrame()
    ref = datetime(2026, 9, 15, 16, 47, tzinfo=MADRID)
    out = _filter_non_eod_batch(df, ref)
    assert out.empty


def test_filter_no_reference_date_returns_unchanged():
    df = _make_batch(["AAPL"])
    out = _filter_non_eod_batch(df, None)
    assert len(out) == 3


def test_filter_past_last_date_kept():
    ref = datetime(2026, 9, 15, 16, 47, tzinfo=MADRID)
    df = _make_batch(["AAPL"], last_date="2026-09-14")
    out = _filter_non_eod_batch(df, ref)
    assert len(out) == 3
    assert out.index[-1] == pd.Timestamp("2026-09-14")


def test_filter_us_market_open_removes_last_row():
    ref = datetime(2026, 9, 15, 16, 47, tzinfo=MADRID)
    df = _make_batch(["AAPL", "MSFT", "GOOG"])
    out = _filter_non_eod_batch(df, ref)
    assert len(out) == 2
    assert out.index[-1] == pd.Timestamp("2026-09-14")


def test_filter_us_market_closed_keeps_last_row():
    ref = datetime(2026, 9, 15, 23, 0, tzinfo=MADRID)
    df = _make_batch(["AAPL", "MSFT"])
    out = _filter_non_eod_batch(df, ref)
    assert len(out) == 3


def test_filter_uk_market_open_removes_last_row():
    ref = datetime(2026, 9, 15, 15, 0, tzinfo=MADRID)
    df = _make_batch(["HSBA.L", "BP.L"])
    out = _filter_non_eod_batch(df, ref)
    assert len(out) == 2


def test_filter_uk_market_closed_keeps_last_row():
    ref = datetime(2026, 9, 15, 18, 0, tzinfo=MADRID)
    df = _make_batch(["HSBA.L"])
    out = _filter_non_eod_batch(df, ref)
    assert len(out) == 3


def test_filter_mixed_batch_us_open_uk_closed():
    """Batch mixto -> conservador, elimina."""
    ref = datetime(2026, 9, 15, 16, 47, tzinfo=MADRID)
    df = _make_batch(["AAPL", "HSBA.L"])
    out = _filter_non_eod_batch(df, ref)
    assert len(out) == 2


def test_filter_unknown_market_forces_removal():
    """UNKNOWN no elegible -> eliminar."""
    ref = datetime(2026, 9, 15, 23, 0, tzinfo=MADRID)
    df = _make_batch(["AAPL", "XYZ.ZZ"])
    out = _filter_non_eod_batch(df, ref)
    assert len(out) == 2


def test_filter_unknown_market_only_removes():
    ref = datetime(2026, 9, 15, 23, 0, tzinfo=MADRID)
    df = _make_batch(["ZZZ.ZZ"])
    out = _filter_non_eod_batch(df, ref)
    assert len(out) == 2


def test_filter_weekend_no_removal():
    ref = datetime(2026, 9, 19, 16, 0, tzinfo=MADRID)
    df = _make_batch(["AAPL"], last_date="2026-09-19")
    out = _filter_non_eod_batch(df, ref)
    assert len(out) == 3


def test_filter_us_holiday_no_removal():
    ref = datetime(2026, 9, 7, 23, 0, tzinfo=MADRID)
    df = _make_batch(["AAPL"], last_date="2026-09-07")
    out = _filter_non_eod_batch(df, ref)
    assert len(out) == 3
