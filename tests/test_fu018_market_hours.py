# -*- coding: utf-8 -*-
"""Tests FU-018 modelo temporal minimo (EOD eligibility)."""
from datetime import datetime, date
from zoneinfo import ZoneInfo

import pytest

from src.market_hours import (
    UnknownMarketError,
    is_trading_session,
    get_session_close,
    is_session_closed,
)
from src.instrument_registry import get_market


MADRID = ZoneInfo("Europe/Madrid")


# --- get_market ---

def test_get_market_us_no_suffix():
    assert get_market("AAPL") == "US_EQUITY"
    assert get_market("BRK-B") == "US_EQUITY"


def test_get_market_lse():
    assert get_market("HSBA.L") == "LSE"


def test_get_market_xetra():
    assert get_market("SAP.DE") == "XETRA"


def test_get_market_bme():
    assert get_market("SAN.MC") == "BME"


def test_get_market_euronext():
    assert get_market("AIR.PA") == "EURONEXT"
    assert get_market("ASML.AS") == "EURONEXT"
    assert get_market("UCG.MI") == "EURONEXT"


def test_get_market_unknown_suffix():
    assert get_market("XYZ.ZZ") == "UNKNOWN"


# --- is_trading_session ---

def test_is_trading_session_us_weekday():
    assert is_trading_session("US_EQUITY", date(2026, 9, 14)) is True


def test_is_trading_session_us_weekend():
    assert is_trading_session("US_EQUITY", date(2026, 9, 12)) is False


def test_is_trading_session_us_holiday_labor_day():
    assert is_trading_session("US_EQUITY", date(2026, 9, 7)) is False


def test_is_trading_session_xetra_weekend():
    assert is_trading_session("XETRA", date(2026, 9, 13)) is False


def test_is_trading_session_unknown_market():
    with pytest.raises(UnknownMarketError):
        is_trading_session("MARS", date(2026, 9, 14))


# --- get_session_close ---

def test_get_session_close_us():
    dt = get_session_close("US_EQUITY", date(2026, 9, 14))
    assert dt.tzinfo is not None
    assert dt.hour == 16
    assert dt.minute == 0
    assert "America/New_York" in str(dt.tzinfo)


def test_get_session_close_xetra():
    dt = get_session_close("XETRA", date(2026, 9, 14))
    assert dt.hour == 17
    assert dt.minute == 30
    assert "Europe/Berlin" in str(dt.tzinfo)


def test_get_session_close_non_trading_day_raises():
    with pytest.raises(ValueError):
        get_session_close("US_EQUITY", date(2026, 9, 12))


def test_get_session_close_holiday_raises():
    with pytest.raises(ValueError):
        get_session_close("US_EQUITY", date(2026, 9, 7))


# --- is_session_closed ---

def test_is_session_closed_naive_reference_raises():
    with pytest.raises(ValueError):
        is_session_closed(
            "US_EQUITY",
            date(2026, 9, 14),
            datetime(2026, 9, 15, 10, 0),
        )


def test_is_session_closed_past_session():
    ref = datetime(2026, 9, 15, 10, 0, tzinfo=MADRID)
    assert is_session_closed("US_EQUITY", date(2026, 9, 14), ref) is True


def test_is_session_closed_future_session():
    ref = datetime(2026, 9, 15, 10, 0, tzinfo=MADRID)
    assert is_session_closed("US_EQUITY", date(2026, 9, 16), ref) is False


def test_is_session_closed_same_day_before_close():
    ref = datetime(2026, 9, 15, 15, 0, tzinfo=MADRID)
    assert is_session_closed("US_EQUITY", date(2026, 9, 15), ref) is False


def test_is_session_closed_same_day_after_close():
    ref = datetime(2026, 9, 15, 23, 0, tzinfo=MADRID)
    assert is_session_closed("US_EQUITY", date(2026, 9, 15), ref) is True


def test_is_session_closed_xetra_same_day_1647_madrid():
    ref = datetime(2026, 9, 15, 16, 47, tzinfo=MADRID)
    assert is_session_closed("XETRA", date(2026, 9, 15), ref) is False


def test_is_session_closed_xetra_same_day_1900_madrid():
    ref = datetime(2026, 9, 15, 19, 0, tzinfo=MADRID)
    assert is_session_closed("XETRA", date(2026, 9, 15), ref) is True


def test_is_session_closed_lse_same_day_1500_madrid():
    ref = datetime(2026, 9, 15, 15, 0, tzinfo=MADRID)
    assert is_session_closed("LSE", date(2026, 9, 15), ref) is False


def test_is_session_closed_lse_same_day_1800_madrid():
    ref = datetime(2026, 9, 15, 18, 0, tzinfo=MADRID)
    assert is_session_closed("LSE", date(2026, 9, 15), ref) is True


def test_is_session_closed_non_trading_day_raises():
    ref = datetime(2026, 9, 15, 10, 0, tzinfo=MADRID)
    with pytest.raises(ValueError):
        is_session_closed("US_EQUITY", date(2026, 9, 12), ref)


def test_is_session_closed_unknown_market_raises():
    ref = datetime(2026, 9, 15, 10, 0, tzinfo=MADRID)
    with pytest.raises(UnknownMarketError):
        is_session_closed("MARS", date(2026, 9, 14), ref)


# --- DST ---

def test_dst_fall_back_europe():
    close = get_session_close("XETRA", date(2026, 10, 26))
    assert close.hour == 17
    assert close.minute == 30
    assert "Europe/Berlin" in str(close.tzinfo)


def test_dst_fall_back_us():
    close = get_session_close("US_EQUITY", date(2026, 11, 2))
    assert close.hour == 16
    assert close.minute == 0
