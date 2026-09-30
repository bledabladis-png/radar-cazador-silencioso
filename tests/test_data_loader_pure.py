# -*- coding: utf-8 -*-
"""D18 (2026-09-30): tests de las funciones puras de src/data_loader.

Complementa test_data_loader_cache_postprocess.py (que cubre
_postprocess_market_data via monkeypatch) y test_fu018_* (que cubren
filtros de providers, no de data_loader).

Cubre:
- _is_equity_ticker: distingue equity USA de indices, futuros, FX, DXY.
- _ticker_list: lee MARKET_TICKERS + etf_holdings.csv.
- _check_khuerfano: deteccion pasiva de Close NaN en expected_session.

No cubre download_market_data (288-372): es orquestacion de red con
DataRouter + BackupProvider. Se cubre por runs reales (E2E) y por
health_check en cron.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from src import data_loader as dl


# ---------- _is_equity_ticker ----------
@pytest.mark.parametrize("ticker", [
    "AAPL", "MSFT", "SPY", "IWM", "NVDA", "XLK", "ASML.AS", "SAP.DE",
])
def test_is_equity_ticker_positivos(ticker):
    assert dl._is_equity_ticker(ticker) is True


@pytest.mark.parametrize("ticker", [
    "^GSPC", "^VIX", "^TNX", "^FVX", "^FTSE", "^GDAXI", "^IBEX",
    "^STOXX50E", "^SPGSCI",
])
def test_is_equity_ticker_indices_false(ticker):
    assert dl._is_equity_ticker(ticker) is False


@pytest.mark.parametrize("ticker", ["CL=F", "BZ=F", "NG=F", "GC=F", "HG=F"])
def test_is_equity_ticker_futuros_false(ticker):
    assert dl._is_equity_ticker(ticker) is False


@pytest.mark.parametrize("ticker", ["EURUSD=X", "USDJPY=X", "USDCNY=X"])
def test_is_equity_ticker_fx_false(ticker):
    assert dl._is_equity_ticker(ticker) is False


def test_is_equity_ticker_dxy_false():
    assert dl._is_equity_ticker("DX-Y.NYB") is False


def test_is_equity_ticker_no_str():
    """Fuerza str() y no crashea."""
    assert dl._is_equity_ticker(None) is True  # "None" no empieza por ^ ni acaba =F
    assert dl._is_equity_ticker(123) is True


# ---------- _ticker_list ----------
def test_ticker_list_sin_etf_holdings():
    """Sin CSV de holdings -> solo MARKET_TICKERS."""
    with patch("pandas.read_csv", side_effect=FileNotFoundError):
        result = dl._ticker_list()
    assert isinstance(result, list)
    assert len(result) > 0  # MARKET_TICKERS siempre tiene tickers


def test_ticker_list_con_etf_holdings():
    """Con CSV -> anade tickers normalizados."""
    fake_holdings = pd.DataFrame({"etf": ["XLK", "XLF"], "ticker": ["NVDA", "JPM"]})
    with patch("pandas.read_csv", return_value=fake_holdings):
        result = dl._ticker_list()
    assert "NVDA" in result
    assert "JPM" in result


def test_ticker_list_sin_columna_ticker():
    """CSV sin columna ticker -> ignorado sin crash."""
    fake_holdings = pd.DataFrame({"etf": ["XLK"], "isin": ["US123"]})
    with patch("pandas.read_csv", return_value=fake_holdings):
        result = dl._ticker_list()
    assert isinstance(result, list)


def test_ticker_list_deduplica():
    """Tickers duplicados entre MARKET_TICKERS y holdings -> set."""
    fake_holdings = pd.DataFrame({"etf": ["XLK"], "ticker": ["AAPL"]})
    with patch("pandas.read_csv", return_value=fake_holdings):
        result = dl._ticker_list()
    # set() garantiza unicidad
    assert len(result) == len(set(result))


# ---------- _check_khuerfano ----------
def _make_batch(tickers, session, values):
    """DataFrame MultiIndex con ('Close', ticker) en el indice session."""
    idx = pd.DatetimeIndex([session])
    cols = pd.MultiIndex.from_tuples([("Close", t) for t in tickers])
    return pd.DataFrame([values], index=idx, columns=cols)


def test_khuerfano_expected_none():
    batch_df = _make_batch(["AAPL"], "2026-09-29", [100.0])
    assert dl._check_khuerfano(batch_df, ["AAPL"], None) == []


def test_khuerfano_data_batch_none():
    assert dl._check_khuerfano(None, ["AAPL"], "2026-09-29") == []


def test_khuerfano_data_batch_vacio():
    assert dl._check_khuerfano(pd.DataFrame(), ["AAPL"], "2026-09-29") == []


def test_khuerfano_ticker_no_en_columnas():
    batch_df = _make_batch(["AAPL"], "2026-09-29", [100.0])
    # "MSFT" no esta en columnas -> ignorado
    assert dl._check_khuerfano(batch_df, ["MSFT"], "2026-09-29") == []


def test_khuerfano_ticker_con_nan_en_expected():
    batch_df = _make_batch(["AAPL", "MSFT"], "2026-09-29", [float("nan"), 200.0])
    out = dl._check_khuerfano(batch_df, ["AAPL", "MSFT"], "2026-09-29")
    assert out == ["AAPL"]


def test_khuerfano_todos_presentes():
    batch_df = _make_batch(["AAPL", "MSFT"], "2026-09-29", [100.0, 200.0])
    assert dl._check_khuerfano(batch_df, ["AAPL", "MSFT"], "2026-09-29") == []


def test_khuerfano_expected_no_en_indice():
    """expected_session no esta en el indice -> no detecta nada."""
    batch_df = _make_batch(["AAPL"], "2026-09-29", [100.0])
    assert dl._check_khuerfano(batch_df, ["AAPL"], "2026-10-01") == []
