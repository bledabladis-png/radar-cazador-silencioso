# -*- coding: utf-8 -*-
"""Cierres excepcionales NYSE (luto nacional).

2018-12-05: George H.W. Bush.
2025-01-09: Jimmy Carter.

NYSE cerro. is_market_day debe devolver False.
"""
from datetime import date

from src.market_calendar import is_market_day, last_expected_market_date


def test_bush_mourning_2018_12_05():
    assert is_market_day(date(2018, 12, 5)) is False


def test_carter_mourning_2025_01_09():
    assert is_market_day(date(2025, 1, 9)) is False


def test_last_expected_market_date_evita_2025_01_09():
    """El dia de luto no puede ser la fecha efectiva esperada."""
    # 2025-01-10 es viernes bursatil. last_expected con hora tarde
    # debe devolver 2025-01-10, nunca 2025-01-09.
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ref = datetime(2025, 1, 10, 23, 30, tzinfo=ZoneInfo("Europe/Madrid"))
    assert last_expected_market_date(ref) == date(2025, 1, 10)