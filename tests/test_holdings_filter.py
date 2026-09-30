# -*- coding: utf-8 -*-
"""D11 (2026-09-30): tests del filtro de tickers de holdings."""
import pandas as pd
import pytest

from src.holdings_filter import is_valid_holding_ticker


@pytest.mark.parametrize("ticker", [
    "AAPL", "NVDA", "GOOGL", "ASML.AS", "SAP.DE", "BRK-B", "BRK.B",
    "MT.AS", "SAN.MC", "AIR.PA", "ISF.L", "IWM", "SPY", "QQQ",
    "MUV2.DE", "DB1.DE", "BH A", "MOG A", "CRD A", "GEF B",
])
def test_validos(ticker):
    assert is_valid_holding_ticker(ticker)


@pytest.mark.parametrize("ticker", [
    "-", "", " ", "2602335D", "IXAU6", "XARU6", "IXDU6", "IXTU6",
    "IXIU6", "IXRU6", "IXPU6", "IXCU6", "IXYU6", "IXSU6", "XASU6",
    "US DOLLAR", "ARCELLX INC CVR", "999USDZ92", "ADI394XJ2",
    "P5N994", "1ABC", "AB.CD.EF", "AB--CD", "AB..CD", "US DOLLAR",
])
def test_invalidos(ticker):
    assert not is_valid_holding_ticker(ticker)


def test_tipo_no_str():
    assert not is_valid_holding_ticker(None)
    assert not is_valid_holding_ticker(123)
    assert not is_valid_holding_ticker([])


def test_whitespace_alrededor():
    assert is_valid_holding_ticker("  aapl  ")
    assert is_valid_holding_ticker("\tbrk-b\n")


def test_regresion_d11_tickers_residuales_historicos():
    """Regresion: los tickers residuales detectados en D11 quedan fuera."""
    invalidos_historicos = [
        "-", "IXTU6", "IXYU6", "IXRU6", "IXIU6", "IXDU6",
        "IXSU6", "XARU6", "XASU6", "2602335D", "P5N994",
        "999USDZ92", "ADI394XJ2",
    ]
    for t in invalidos_historicos:
        assert not is_valid_holding_ticker(t), "aceptado invalido: {}".format(t)


def test_csv_etf_holdings_sin_invalidos():
    """etf_holdings.csv en disco no contiene tickers invalidos."""
    df = pd.read_csv("data/etf_holdings.csv")
    invalidos = [t for t in df["ticker"].astype(str)
                 if not is_valid_holding_ticker(t)]
    assert not invalidos, "tickers invalidos: {}".format(sorted(set(invalidos)))


def test_csv_index_holdings_sin_invalidos():
    """index_holdings.csv en disco no contiene tickers invalidos."""
    df = pd.read_csv("data/index_holdings.csv")
    invalidos = [t for t in df["ticker"].astype(str)
                 if not is_valid_holding_ticker(t)]
    assert not invalidos, "tickers invalidos: {}".format(sorted(set(invalidos)))


def test_data_loader_sin_lista_negra():
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "data_loader.py"
    text = src.read_text(encoding="utf-8")
    assert "INVALID_TICKERS" not in text, "lista negra residual"
