# -*- coding: utf-8 -*-
"""Tests de regresion para XetraProvider (bug 2026-10-25).

El provider fallaba con `'datetime.date' object has no attribute 'date'`
cuando reference_date llegaba como datetime.date (no Timestamp). La
normalizacion con _as_date() lo resuelve para date, datetime, Timestamp,
string ISO y None.
"""
from datetime import date, datetime

import pandas as pd
import pytest

from data.providers.xetra_provider import XetraProvider, _as_date


def _sample_df():
    return pd.DataFrame({
        "date": [pd.Timestamp("2026-10-22"), pd.Timestamp("2026-10-23")],
        "close": [100.0, 101.0],
    })


def test_as_date_none_devuelve_hoy():
    result = _as_date(None)
    assert isinstance(result, date)


def test_as_date_desde_date():
    d = date(2026, 10, 23)
    assert _as_date(d) == d


def test_as_date_desde_datetime():
    dt = datetime(2026, 10, 23, 15, 30)
    assert _as_date(dt) == date(2026, 10, 23)


def test_as_date_desde_timestamp():
    ts = pd.Timestamp("2026-10-23 15:30")
    assert _as_date(ts) == date(2026, 10, 23)


def test_as_date_desde_string_iso():
    assert _as_date("2026-10-23") == date(2026, 10, 23)


@pytest.mark.parametrize("ref", [
    date(2026, 10, 24),
    datetime(2026, 10, 24, 10, 0),
    pd.Timestamp("2026-10-24"),
    "2026-10-24",
    None,
])
def test_filter_non_eod_last_row_acepta_cualquier_tipo_ref(ref):
    """El bug: reference_date.date() fallaba si ya era date."""
    p = XetraProvider()
    try:
        result = p._filter_non_eod_last_row(_sample_df(), "TEST.DE", reference_date=ref)
    except AttributeError as e:
        pytest.fail(f"AttributeError con ref={ref!r}: {e}")
    assert result is not None
