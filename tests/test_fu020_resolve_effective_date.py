# -*- coding: utf-8 -*-
"""Tests FU-020: resolve_effective_date."""
import numpy as np
import pandas as pd
import pytest

from src.effective_date import resolve_effective_date


def _make_prices(dates, tickers, presence):
    """presence: dict ticker -> lista de True/False por fecha.
    True = valor, False = NaN."""
    data = {}
    for t in tickers:
        vals = [100.0 if present else np.nan for present in presence[t]]
        data[t] = vals
    return pd.DataFrame(data, index=pd.to_datetime(dates))


# --- Casos basicos ---

def test_all_covered_returns_last_date():
    dates = ["2026-09-11", "2026-09-14", "2026-09-15"]
    tickers = ["A", "B", "C"]
    presence = {t: [True, True, True] for t in tickers}
    prices = _make_prices(dates, tickers, presence)

    r = resolve_effective_date(prices, tickers, min_coverage=0.90)
    assert r["status"] == "OK"
    assert pd.Timestamp(r["date"]) == pd.Timestamp("2026-09-15")
    assert r["n_eligible"] == 3
    assert r["n_observed"] == 3
    assert r["coverage"] == 1.0
    assert r["lag_days"] == 0


def test_partial_coverage_returns_previous_date():
    """Fila superior parcial (33%) -> retrocede a la anterior (100%)."""
    dates = ["2026-09-11", "2026-09-14", "2026-09-15"]
    tickers = ["A", "B", "C"]
    presence = {
        "A": [True, True, True],
        "B": [True, True, False],
        "C": [True, True, False],
    }
    prices = _make_prices(dates, tickers, presence)

    r = resolve_effective_date(prices, tickers, min_coverage=0.90)
    assert r["status"] == "OK"
    assert pd.Timestamp(r["date"]) == pd.Timestamp("2026-09-14")
    assert r["lag_days"] == 1


def test_no_valid_date_returns_insufficient():
    dates = ["2026-09-14", "2026-09-15"]
    tickers = ["A", "B", "C"]
    presence = {
        "A": [False, True],
        "B": [False, False],
        "C": [False, False],
    }
    prices = _make_prices(dates, tickers, presence)

    r = resolve_effective_date(prices, tickers, min_coverage=0.90)
    assert r["status"] == "INSUFFICIENT_COVERAGE"
    assert r["date"] is None
    assert r["n_observed"] == 0
    assert r["coverage"] == 0.0


# --- Validaciones del universo elegible ---

def test_duplicates_in_eligible_are_deduped():
    dates = ["2026-09-15"]
    tickers = ["A", "B"]
    presence = {"A": [True], "B": [True]}
    prices = _make_prices(dates, tickers, presence)

    r = resolve_effective_date(prices, ["A", "B", "A"], min_coverage=0.90)
    assert r["n_eligible"] == 2
    assert r["coverage"] == 1.0


def test_missing_columns_count_as_not_observed():
    """Elegible sin columna en prices -> no observado, cuenta en denominador."""
    dates = ["2026-09-15"]
    tickers = ["A", "B"]
    presence = {"A": [True], "B": [True]}
    prices = _make_prices(dates, tickers, presence)

    r = resolve_effective_date(prices, ["A", "B", "C"], min_coverage=0.66)
    assert r["n_eligible"] == 3
    assert r["n_observed"] == 2
    assert r["coverage"] == pytest.approx(2 / 3)


def test_empty_eligible_returns_insufficient():
    dates = ["2026-09-15"]
    tickers = ["A"]
    presence = {"A": [True]}
    prices = _make_prices(dates, tickers, presence)

    r = resolve_effective_date(prices, [], min_coverage=0.90)
    assert r["status"] == "INSUFFICIENT_COVERAGE"
    assert r["n_eligible"] == 0


def test_empty_prices_returns_insufficient():
    prices = pd.DataFrame()
    r = resolve_effective_date(prices, ["A"], min_coverage=0.90)
    assert r["status"] == "INSUFFICIENT_COVERAGE"
    assert r["date"] is None


def test_no_present_columns_returns_insufficient():
    dates = ["2026-09-15"]
    prices = pd.DataFrame({"X": [100.0]}, index=pd.to_datetime(dates))

    r = resolve_effective_date(prices, ["A", "B"], min_coverage=0.90)
    assert r["status"] == "INSUFFICIENT_COVERAGE"
    assert r["n_observed"] == 0


# --- Lag ---

def test_lag_days_zero_when_same_date():
    dates = ["2026-09-15"]
    prices = _make_prices(dates, ["A"], {"A": [True]})
    r = resolve_effective_date(prices, ["A"], min_coverage=0.90)
    assert r["lag_days"] == 0


def test_lag_days_positive_when_previous_date():
    dates = ["2026-09-11", "2026-09-14", "2026-09-15"]
    presence = {"A": [True, True, False], "B": [True, True, False]}
    prices = _make_prices(dates, ["A", "B"], presence)
    r = resolve_effective_date(prices, ["A", "B"], min_coverage=0.90)
    assert r["lag_days"] == 1


# --- Casos del auditor: denominador contextual ---

def test_usa_only_universe_accepts_us_only_date():
    """eligible = USA. 262/262 observados -> 100% -> acepta 15/09."""
    dates = ["2026-09-14", "2026-09-15"]
    usa = [f"U{i}" for i in range(5)]
    presence = {t: [True, True] for t in usa}
    prices = _make_prices(dates, usa, presence)

    r = resolve_effective_date(prices, usa, min_coverage=0.90)
    assert r["status"] == "OK"
    assert pd.Timestamp(r["date"]) == pd.Timestamp("2026-09-15")
    assert r["coverage"] == 1.0
    assert r["lag_days"] == 0


def test_global_universe_rejects_us_only_date():
    """eligible = USA + Europa. 5 USA OK, 5 Europa sin dato -> 50% global."""
    dates = ["2026-09-14", "2026-09-15"]
    usa = [f"U{i}" for i in range(5)]
    eu = [f"E{i}" for i in range(5)]
    presence = {t: [True, True] for t in usa}
    presence.update({t: [True, False] for t in eu})
    prices = _make_prices(dates, usa + eu, presence)

    r = resolve_effective_date(prices, usa + eu, min_coverage=0.90)
    assert r["status"] == "OK"
    # 10 elegibles, 5 observados en 15/09 -> 50% < 90% -> retrocede a 14/09
    assert pd.Timestamp(r["date"]) == pd.Timestamp("2026-09-14")
    assert r["coverage"] == 1.0
    assert r["lag_days"] == 1


def test_denominator_is_eligible_not_present():
    """Si eligible = 10 pero solo 8 columnas existen, denominador es 10."""
    dates = ["2026-09-15"]
    tickers = [f"T{i}" for i in range(8)]
    presence = {t: [True] for t in tickers}
    prices = _make_prices(dates, tickers, presence)

    eligible = [f"T{i}" for i in range(10)]  # 2 tickers que no existen
    r = resolve_effective_date(prices, eligible, min_coverage=0.50)
    assert r["status"] == "OK"
    assert r["n_eligible"] == 10
    assert r["n_observed"] == 8
    assert r["coverage"] == pytest.approx(0.8)
