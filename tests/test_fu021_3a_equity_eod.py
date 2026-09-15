# -*- coding: utf-8 -*-
"""Tests FU-021-3A: filtro EQUITY_EOD en market_data."""
import numpy as np
import pandas as pd

from datetime import datetime
from zoneinfo import ZoneInfo

from src.data_loader import (
    _is_equity_ticker,
    _trim_market_data_to_equity_eod,
    _filter_non_eod_equity,
)
from src.effective_date import resolve_effective_date


# --- _is_equity_ticker ---

def test_is_equity_plain_ticker():
    assert _is_equity_ticker("AAPL") is True
    assert _is_equity_ticker("SPY") is True
    assert _is_equity_ticker("XLE") is True


def test_is_equity_excludes_indices():
    for t in ["^GSPC", "^DJI", "^NDX", "^RUT", "^VIX", "^VIX3M", "^VXN",
              "^TNX", "^FVX", "^FTSE", "^GDAXI", "^IBEX", "^STOXX50E", "^SPGSCI"]:
        assert _is_equity_ticker(t) is False, f"{t} deberia ser NO equity"


def test_is_equity_excludes_futures():
    for t in ["CL=F", "BZ=F", "NG=F", "GC=F", "HG=F"]:
        assert _is_equity_ticker(t) is False, f"{t} deberia ser NO equity"


def test_is_equity_excludes_fx():
    for t in ["EURUSD=X", "USDJPY=X", "USDCNY=X"]:
        assert _is_equity_ticker(t) is False, f"{t} deberia ser NO equity"


def test_is_equity_excludes_dxy():
    assert _is_equity_ticker("DX-Y.NYB") is False


# --- _trim_market_data_to_equity_eod ---

def _make_market_data(dates, equity_presence, other_presence=None):
    """Construye DataFrame MultiIndex (field, ticker).

    equity_presence: dict ticker_equity -> lista [True/False] por fecha
    other_presence:  dict ticker_other  -> lista [True/False] por fecha (opcional)
    """
    data = {}
    def _add(t, presence):
        closes = [100.0 + i * 0.1 if p else np.nan for i, p in enumerate(presence)]
        for field in ("Open", "High", "Low", "Close", "Volume"):
            if field == "Close":
                data[(field, t)] = closes
            else:
                data[(field, t)] = [v + 1 if not np.isnan(v) else np.nan for v in closes]
    for t, pres in equity_presence.items():
        _add(t, pres)
    if other_presence:
        for t, pres in other_presence.items():
            _add(t, pres)
    df = pd.DataFrame(data, index=pd.to_datetime(dates))
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_trim_no_equity_returns_unchanged():
    """Sin equities en el DataFrame -> sin trim, meta=None."""
    dates = ["2026-09-14", "2026-09-15"]
    df = _make_market_data(dates, {}, {"^GSPC": [True, True]})
    out, meta = _trim_market_data_to_equity_eod(df)
    assert meta is None
    assert len(out) == 2


def test_trim_usa_open_trims_to_previous_close():
    """USA abierto en ultima fecha (equities NaN) -> trim a fecha previa."""
    dates = ["2026-09-11", "2026-09-14", "2026-09-15"]
    # 30 equities con NaN en la fila del 15/09 (USA abierto)
    equity_presence = {f"E{i}": [True, True, False] for i in range(30)}
    df = _make_market_data(dates, equity_presence)

    out, meta = _trim_market_data_to_equity_eod(df)
    assert meta['status'] == 'OK'
    assert pd.Timestamp(meta['date']) == pd.Timestamp("2026-09-14")
    assert meta['lag_days'] == 1
    assert len(out) == 2
    assert out.index[-1] == pd.Timestamp("2026-09-14")


def test_trim_usa_closed_keeps_last_row():
    """USA cerrado (equities con dato) -> no trim."""
    dates = ["2026-09-11", "2026-09-14", "2026-09-15"]
    equity_presence = {f"E{i}": [True, True, True] for i in range(30)}
    df = _make_market_data(dates, equity_presence)

    out, meta = _trim_market_data_to_equity_eod(df)
    assert meta['status'] == 'OK'
    assert meta['lag_days'] == 0
    assert len(out) == 3
    assert out.index[-1] == pd.Timestamp("2026-09-15")


def test_trim_excludes_other_instruments_from_coverage():
    """Los no-equity no cuentan para la cobertura, pero se recortan con la fila."""
    dates = ["2026-09-14", "2026-09-15"]
    # 30 equities NaN en 15/09; indices y futuros con dato en 15/09
    equity_presence = {f"E{i}": [True, False] for i in range(30)}
    other_presence = {
        "^GSPC": [True, True],
        "CL=F": [True, True],
        "DX-Y.NYB": [True, True],
    }
    df = _make_market_data(dates, equity_presence, other_presence)

    out, meta = _trim_market_data_to_equity_eod(df)
    assert meta['status'] == 'OK'
    assert pd.Timestamp(meta['date']) == pd.Timestamp("2026-09-14")
    # La fila del 15/09 completa desaparece, incluidos indices/futuros
    assert len(out) == 1
    assert out.index[-1] == pd.Timestamp("2026-09-14")


def test_trim_insufficient_coverage_no_trim():
    """Cobertura < 90% en todas las fechas -> no trim, meta con status INSUFFICIENT."""
    dates = ["2026-09-14", "2026-09-15"]
    # 30 equities, solo 20 con dato (66%)
    equity_presence = {f"E{i}": [i < 20, i < 20] for i in range(30)}
    df = _make_market_data(dates, equity_presence)

    out, meta = _trim_market_data_to_equity_eod(df)
    assert meta['status'] == 'INSUFFICIENT_COVERAGE'
    assert len(out) == 2


def test_trim_empty_df():
    df = pd.DataFrame()
    out, meta = _trim_market_data_to_equity_eod(df)
    assert out.empty
    assert meta is None


def test_trim_lag_zero_does_not_trim():
    """Si effective == requested, no hay trim."""
    dates = ["2026-09-15"]
    equity_presence = {f"E{i}": [True] for i in range(30)}
    df = _make_market_data(dates, equity_presence)

    out, meta = _trim_market_data_to_equity_eod(df)
    assert meta['lag_days'] == 0
    assert len(out) == 1


# --- FU-021-3A correction v2: filtro sesion/EOD equity-only ---


def _nyse_open_ref():
    """2026-09-15 13:00 ET = 19:00 Madrid. Sesion USA abierta."""
    return datetime(2026, 9, 15, 19, 0, tzinfo=ZoneInfo("Europe/Madrid"))


def _nyse_closed_ref():
    """2026-09-15 23:00 Madrid = 17:00 ET. Sesion USA cerrada."""
    return datetime(2026, 9, 15, 23, 0, tzinfo=ZoneInfo("Europe/Madrid"))


def test_filter_non_eod_equity_usa_open_trims():
    """USA abierta: equity con dato intradia en 15/09 -> trim a 14/09."""
    dates = ["2026-09-11", "2026-09-14", "2026-09-15"]
    equity_presence = {f"E{i}": [True, True, True] for i in range(30)}
    df = _make_market_data(dates, equity_presence)

    out, info = _filter_non_eod_equity(df, _nyse_open_ref())
    assert info["n_equity"] == 30
    assert info["n_present"] == 30
    assert info["non_eod"] == 30
    assert len(out) == 2
    assert out.index[-1] == pd.Timestamp("2026-09-14")


def test_filter_non_eod_equity_usa_closed_keeps():
    """USA cerrada: equity con dato en 15/09 -> sin trim."""
    dates = ["2026-09-11", "2026-09-14", "2026-09-15"]
    equity_presence = {f"E{i}": [True, True, True] for i in range(30)}
    df = _make_market_data(dates, equity_presence)

    out, info = _filter_non_eod_equity(df, _nyse_closed_ref())
    assert info["n_equity"] == 30
    assert info["non_eod"] == 0
    assert len(out) == 3
    assert out.index[-1] == pd.Timestamp("2026-09-15")


def test_filter_non_eod_equity_ignores_non_equity_only():
    """DataFrame sin equity -> sin cambios, n_equity=0."""
    dates = ["2026-09-14", "2026-09-15"]
    other = {"^GSPC": [True, True], "CL=F": [True, True]}
    df = _make_market_data(dates, {}, other)

    out, info = _filter_non_eod_equity(df, _nyse_open_ref())
    assert info["n_equity"] == 0
    assert len(out) == 2
    assert out.index[-1] == pd.Timestamp("2026-09-15")


def test_filter_non_eod_equity_100pct_intraday_still_trims():
    """Test clave del auditor: 100% cobertura NO implica EOD.

    Reproduce el caso del run 2026-09-15T17:04Z: 30 equities con
    precio vivo (no NaN) en la fila intradia. Cobertura = 100%.
    Con USA abierta, el filtro DEBE eliminar esa fila.
    """
    dates = ["2026-09-14", "2026-09-15"]
    equity_presence = {f"E{i}": [True, True] for i in range(30)}
    df = _make_market_data(dates, equity_presence)

    # Cobertura 100% en la ultima fila
    close_cols = [c for c in df.columns if c[0] == "Close"]
    coverage = df[close_cols].notna().sum(axis=1) / len(close_cols)
    assert coverage.iloc[-1] == 1.0

    out, info = _filter_non_eod_equity(df, _nyse_open_ref())
    assert info["non_eod"] == 30
    assert len(out) == 1
    assert out.index[-1] == pd.Timestamp("2026-09-14")


def test_reference_lag_differs_from_function_lag():
    """Separacion conceptual: function_lag=0, reference_lag=1.

    Tras el filtro EOD, resolve_effective_date opera sobre datos ya
    temporalmente coherentes: requested == effective == 14/09 ->
    function_lag = 0. Pero reference_date es 15/09 -> reference_lag = 1.
    Ambos valores coexisten y no deben fusionarse.
    """
    dates = ["2026-09-14", "2026-09-15"]
    equity_presence = {f"E{i}": [True, True] for i in range(30)}
    df = _make_market_data(dates, equity_presence)

    reference_date = _nyse_open_ref()
    df_filtered, _ = _filter_non_eod_equity(df, reference_date)

    # resolve_effective_date sobre datos ya filtrados
    eligible = [c[1] for c in df_filtered.columns if c[0] == "Close"]
    meta = resolve_effective_date(df_filtered, eligible, min_coverage=0.90)

    assert meta["status"] == "OK"
    assert pd.Timestamp(meta["date"]) == pd.Timestamp("2026-09-14")
    assert pd.Timestamp(meta["requested_date"]) == pd.Timestamp("2026-09-14")
    assert meta["lag_days"] == 0  # function_lag

    ref_lag = (reference_date.date()
               - pd.Timestamp(meta["date"]).date()).days
    assert ref_lag == 1  # reference_lag
    assert ref_lag != meta["lag_days"]
