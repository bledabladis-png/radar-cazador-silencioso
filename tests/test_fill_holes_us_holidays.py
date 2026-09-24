"""Tests del fix F-IAE-HOLIDAY-01 (2026-09-24).

Verifican que _fill_holes_respecting_sessions preserva NaN en tickers
USA cuando la fecha es festivo NYSE en dia laborable, pero mantiene
el comportamiento original (ffill) en fines de semana.

Contexto: los lotes Yahoo mixtos UK+USA provocaban que tickers USA
recibieran ffill desde el viernes previo en festivos NYSE.
"""
import numpy as np
import pandas as pd
import pytest

from src.stock_data_loader import _fill_holes_respecting_sessions


# Festivos NYSE 2026 relevantes:
#  2026-09-07 (Monday) - Labor Day
#  2026-05-25 (Monday) - Memorial Day
#  2026-07-03 (Friday)  - July 4 observed
#
# Fin de semana normal:
#  2026-09-12 (Saturday)
#  2026-09-13 (Sunday)

REFERENCE_DATE = pd.Timestamp("2026-09-14 12:00", tz="Europe/Madrid")


def _mk_df(dates, closes, ticker):
    idx = pd.DatetimeIndex(dates)
    cols = pd.MultiIndex.from_tuples([("Close", ticker)])
    arr = np.array(closes, dtype=float).reshape(-1, 1)
    return pd.DataFrame(arr, index=idx, columns=cols)


# --- Fix F-IAE-HOLIDAY-01 ----------------------------------------------

def test_us_holiday_weekday_preserva_nan():
    """USA en festivo NYSE (lunes) -> NaN preservado, NO ffill."""
    # TEST sin sufijo -> get_market = 'US_EQUITY'
    df = _mk_df(
        ["2026-09-04", "2026-09-07"],
        [100.0, np.nan],
        "TEST",
    )
    df_filled, diag = _fill_holes_respecting_sessions(df, REFERENCE_DATE)
    s = df_filled[("Close", "TEST")]
    assert pd.isna(s.loc["2026-09-07"]), (
        "09-07 es Labor Day (NYSE cerrado); ticker USA debe preservar NaN")
    assert diag["n_preserved_by_us_holiday"] >= 1


def test_us_weekend_si_se_rellena():
    """USA en fin de semana -> ffill normal (comportamiento original)."""
    df = _mk_df(
        ["2026-09-10", "2026-09-11", "2026-09-12"],
        [100.0, np.nan, np.nan],
        "TEST",
    )
    df_filled, diag = _fill_holes_respecting_sessions(df, REFERENCE_DATE)
    s = df_filled[("Close", "TEST")]
    assert s.loc["2026-09-12"] == 100.0, (
        "09-12 es sabado; ffill normal (weekday >= 5)")
    assert diag["n_preserved_by_us_holiday"] == 0


def test_us_multiples_festivos_weekday():
    """Multiples festivos NYSE (Mon/Fri) preservan NaN."""
    df = _mk_df(
        ["2026-06-18", "2026-06-19", "2026-07-02", "2026-07-03"],
        [200.0, np.nan, 300.0, np.nan],
        "TEST",
    )
    df_filled, diag = _fill_holes_respecting_sessions(df, REFERENCE_DATE)
    s = df_filled[("Close", "TEST")]
    assert pd.isna(s.loc["2026-06-19"])
    assert pd.isna(s.loc["2026-07-03"])
    assert diag["n_preserved_by_us_holiday"] >= 2


def test_uk_holiday_usa_no_afecta():
    """Ticker UK (LSE) con valor en festivo NYSE: no se toca."""
    df = _mk_df(
        ["2026-09-04", "2026-09-07"],
        [100.0, 101.0],  # LSE abierto el lunes 09-07
        "TEST.L",
    )
    df_filled, diag = _fill_holes_respecting_sessions(df, REFERENCE_DATE)
    s = df_filled[("Close", "TEST.L")]
    assert s.loc["2026-09-07"] == 101.0, "LSE abierto: valor real no se toca"
    assert diag["n_preserved_by_us_holiday"] == 0

def test_uk_nan_en_festivo_usa_se_rellena():
    """Ticker UK con NaN en festivo NYSE: ffill normal (no preservado)."""
    df = _mk_df(
        ["2026-09-04", "2026-09-07"],
        [100.0, np.nan],
        "TEST.L",
    )
    df_filled, diag = _fill_holes_respecting_sessions(df, REFERENCE_DATE)
    s = df_filled[("Close", "TEST.L")]
    # LSE abierto el 09-07 (Lunes laborable en UK). Si hay NaN, se rellena
    # porque is_market_day(2026-09-07)==False (USA festivo) y el ticker
    # no es US_EQUITY -> cae en ffill.
    assert s.loc["2026-09-07"] == 100.0
    assert diag["n_preserved_by_us_holiday"] == 0


def test_us_en_sesion_nyse_preserva_nan():
    """USA en sesion NYSE normal -> preserva (comportamiento original)."""
    df = _mk_df(
        ["2026-09-10", "2026-09-11"],
        [100.0, np.nan],
        "TEST",
    )
    df_filled, diag = _fill_holes_respecting_sessions(df, REFERENCE_DATE)
    s = df_filled[("Close", "TEST")]
    assert pd.isna(s.loc["2026-09-11"]), "sesion NYSE normal -> preserva"
    assert diag["n_preserved_by_us_holiday"] == 0, (
        "el festivo no cuenta en sesion normal")


def test_mixto_uk_us_mismo_batch():
    """Escenario real: batch mixto UK+USA en festivo NYSE."""
    idx = pd.DatetimeIndex(["2026-09-04", "2026-09-07"])
    cols = pd.MultiIndex.from_tuples([
        ("Close", "TEST.L"),   # UK
        ("Close", "TEST"),     # USA
    ])
    # UK con valor real (LSE abierto), USA NaN (NYSE cerrado)
    arr = np.array([[100.0, 200.0], [101.0, np.nan]])
    df = pd.DataFrame(arr, index=idx, columns=cols)

    df_filled, diag = _fill_holes_respecting_sessions(df, REFERENCE_DATE)
    assert df_filled[("Close", "TEST.L")].loc["2026-09-07"] == 101.0
    assert pd.isna(df_filled[("Close", "TEST")].loc["2026-09-07"]), (
        "USA no debe recibir valor del 09-04 via ffill")


def test_diag_expected_session_presente():
    """El diagnostico incluye expected_session."""
    df = _mk_df(["2026-09-10"], [100.0], "TEST")
    _, diag = _fill_holes_respecting_sessions(df, REFERENCE_DATE)
    assert "expected_session" in diag
    assert "n_preserved_by_us_holiday" in diag