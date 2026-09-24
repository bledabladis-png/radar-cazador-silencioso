"""Tests unitarios de scripts/update_sec_13f.py (F.3.c).

Solo funciones puras: sin red, sin I/O, sin SEC.

Cubre:
  - _is_leap, _last_day_of_month
  - _parse_quarter (validaciones)
  - quarter_to_iso_end (4 trimestres)
  - quarter_to_source_period (formato SEC, cruce de año, bisiesto)
  - latest_published_quarter (regla 60 dias)
  - _pick_base_url (OLD vs NEW)
"""
from __future__ import annotations

from datetime import date

import pytest

from scripts.update_sec_13f import (
    _is_leap,
    _last_day_of_month,
    _parse_quarter,
    quarter_to_iso_end,
    quarter_to_source_period,
    latest_published_quarter,
)
from src.institutional_accumulation.sec_13f.downloader import (
    _pick_base_url,
    SEC_BASE_URL_OLD,
    SEC_BASE_URL_NEW,
)


# --- _is_leap -----------------------------------------------------------

@pytest.mark.parametrize("year,expected", [
    (2024, True), (2025, False), (2000, True), (1900, False), (2100, False),
])
def test_is_leap(year, expected):
    assert _is_leap(year) == expected


# --- _last_day_of_month -------------------------------------------------

@pytest.mark.parametrize("year,month,expected", [
    (2026, 1, 31), (2026, 2, 28), (2024, 2, 29), (2026, 3, 31),
    (2026, 4, 30), (2026, 6, 30), (2026, 9, 30), (2026, 12, 31),
])
def test_last_day_of_month(year, month, expected):
    assert _last_day_of_month(year, month) == expected


# --- _parse_quarter -----------------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("2026Q1", (2026, "Q1")), ("2025Q4", (2025, "Q4")), ("2030Q3", (2030, "Q3")),
])
def test_parse_quarter_valido(q, expected):
    assert _parse_quarter(q) == expected


@pytest.mark.parametrize("q", [
    "2026Q5", "2026-Q1", "26Q1", "", None, "2026X1", "2026Q", "QWERTY",
])
def test_parse_quarter_invalido(q):
    with pytest.raises((ValueError, TypeError)):
        _parse_quarter(q)

# --- quarter_to_iso_end -------------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("2025Q4", "2025-12-31"),
    ("2026Q1", "2026-03-31"),
    ("2026Q2", "2026-06-30"),
    ("2026Q3", "2026-09-30"),
    ("2026Q4", "2026-12-31"),
    ("2024Q2", "2024-06-30"),
])
def test_quarter_to_iso_end(q, expected):
    assert quarter_to_iso_end(q) == expected


# --- quarter_to_source_period -------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("2025Q4", "01dec2025-28feb2026"),
    ("2026Q1", "01mar2026-31may2026"),
    ("2026Q2", "01jun2026-31aug2026"),
    ("2026Q3", "01sep2026-30nov2026"),
    ("2026Q4", "01dec2026-28feb2027"),
    ("2027Q1", "01mar2027-31may2027"),
    # Bisiesto: 2024Q4 -> end_year 2025 (no leap) -> 28feb2025
    ("2024Q4", "01dec2024-28feb2025"),
    # Bisiesto: 2023Q4 -> end_year 2024 (leap) -> 29feb2024
    ("2023Q4", "01dec2023-29feb2024"),
])
def test_quarter_to_source_period(q, expected):
    assert quarter_to_source_period(q) == expected


# --- latest_published_quarter -------------------------------------------

@pytest.mark.parametrize("today,expected", [
    (date(2026, 5, 1), "2025Q4"),   # Q1 2026 aun no pasa 60d (31-mar+60=30-may)
    (date(2026, 6, 30), "2026Q1"),  # 30-may ya paso; Q2 cierra 30-jun no
    (date(2026, 9, 23), "2026Q2"),  # 29-ago paso; Q3 cierra 30-sep no
    (date(2027, 2, 15), "2026Q3"),  # 29-nov-26 paso; Q4 31-dic+60=1-mar-27 no
    (date(2024, 1, 15), "2023Q3"),  # 29-nov-23 paso; 2023Q4 31-dic+60 no
])
def test_latest_published_quarter(today, expected):
    assert latest_published_quarter(today) == expected


# --- _pick_base_url -----------------------------------------------------

@pytest.mark.parametrize("period,expected", [
    ("01dec2025-28feb2026", SEC_BASE_URL_OLD),
    ("01mar2026-31may2026", SEC_BASE_URL_OLD),
    ("01jun2026-31aug2026", SEC_BASE_URL_NEW),
    ("01sep2026-30nov2026", SEC_BASE_URL_NEW),
    ("01dec2026-28feb2027", SEC_BASE_URL_NEW),
    ("01mar2027-31may2027", SEC_BASE_URL_NEW),
])
def test_pick_base_url_por_period(period, expected):
    assert _pick_base_url(period) == expected


def test_pick_base_url_override():
    assert _pick_base_url("01mar2026-31may2026",
                          override="http://custom") == "http://custom"


def test_pick_base_url_vacio_lanza():
    with pytest.raises(ValueError):
        _pick_base_url("")


# --- _prev_quarter / _quarters_range (backfill) ------------------------

from scripts.update_sec_13f import _prev_quarter, _quarters_range


@pytest.mark.parametrize("q,expected", [
    ("2026Q2", "2026Q1"),
    ("2026Q1", "2025Q4"),
    ("2025Q1", "2024Q4"),
    ("2024Q4", "2024Q3"),
])
def test_prev_quarter(q, expected):
    assert _prev_quarter(q) == expected


def test_prev_quarter_invalido_lanza():
    with pytest.raises((ValueError, TypeError)):
        _prev_quarter("INVALID")


def test_quarters_range_cero():
    assert _quarters_range("2026Q2", 0) == ["2026Q2"]


def test_quarters_range_dos():
    assert _quarters_range("2026Q2", 2) == ["2025Q4", "2026Q1", "2026Q2"]


def test_quarters_range_cruce_ano():
    assert _quarters_range("2026Q1", 2) == ["2025Q3", "2025Q4", "2026Q1"]


def test_quarters_range_tres():
    assert _quarters_range("2026Q3", 3) == [
        "2025Q4", "2026Q1", "2026Q2", "2026Q3"
    ]
